import torch
from collections import OrderedDict
from os import path as osp
from tqdm import tqdm
from einops import rearrange

from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.metrics import calculate_metric
from basicsr.utils import get_root_logger, imwrite, tensor2img, tensor2imgs, images_to_gif
from basicsr.utils.registry import MODEL_REGISTRY
import torch.nn.functional as F
from .sr_model import SRModel

@MODEL_REGISTRY.register()
class CodeFormerDirichletVideoModel(SRModel):
    def feed_data(self, data):
        self.gt = data['gt'].to(self.device) # b t c h w
        self.input = data['in'].to(self.device)
        self.lq = data['in'].to(self.device)  # 添加部分
        self.input_large_de = data['in'].to(self.device)
        self.b, self.t = data['gt'].shape[:2]
        # self.input_large_de = data['in_large_de'].to(self.device)

        # 合并b t维度
        self.gt = rearrange(self.gt, "b t ... -> (b t) ...")
        self.input = rearrange(self.input, "b t ... -> (b t) ...")
        self.input_large_de = rearrange(self.input_large_de, "b t ... -> (b t) ...")

        if 'latent_gt' in data:
            self.idx_gt = data['latent_gt'].to(self.device)
            # self.idx_gt = self.idx_gt.view(self.b, -1)
            self.idx_gt = rearrange(self.idx_gt, "b t ... -> (b t) ...")
        else:
            self.idx_gt = None

    def init_training_settings(self):
        logger = get_root_logger()
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        self.scale_adaptive_gan_weight = train_opt.get('scale_adaptive_gan_weight', 0.8)

        # define network net_d
        self.net_d = build_network(self.opt['network_d'])
        self.net_d = self.model_to_device(self.net_d)
        self.print_network(self.net_d)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_d', None)
        if load_path is not None:
            self.load_network(self.net_d, load_path, self.opt['path'].get('strict_load_d', True))

        self.net_g.train()
        self.net_d.train()

        # define losses
        self.cri_pix = None
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        
        self.cri_perceptual = None
        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)

        # add the dir dist KL loss
        self.cri_dirichletKL = None
        if train_opt.get('dirichletKL_opt'):
            self.cri_dirichletKL = build_loss(train_opt['dirichletKL_opt']).to(self.device)

        if train_opt.get('gan_opt'):
            self.cri_gan = build_loss(train_opt['gan_opt']).to(self.device)

        self.fix_generator = train_opt.get('fix_generator', True)
        logger.info(f'fix_generator: {self.fix_generator}')

        self.net_g_start_iter = train_opt.get('net_g_start_iter', 0)
        self.net_d_iters = train_opt.get('net_d_iters', 1)
        self.net_d_start_iter = train_opt.get('net_d_start_iter', 0)

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def calculate_adaptive_weight(self, recon_loss, g_loss, last_layer, disc_weight_max):
        recon_grads = torch.autograd.grad(recon_loss, last_layer, retain_graph=True)[0]
        g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]

        d_weight = torch.norm(recon_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, disc_weight_max).detach()
        return d_weight

    def setup_optimizers(self):
        train_opt = self.opt['train']
        # optimizer g
        optim_params_g = []
        trainable_modules = []
        notrainable_modules = []
        for k, v in self.net_g.named_parameters():
            module_ = '.'.join(k.split('.')[:2])
            if v.requires_grad:
                optim_params_g.append(v)
                if module_ not in trainable_modules:
                    trainable_modules.append(module_)
            else:
                if module_ not in notrainable_modules:
                    notrainable_modules.append(module_)

        logger = get_root_logger()
        for _ in trainable_modules:
            logger.warning(f'{_} will be optimized.')
        for _ in notrainable_modules:
            logger.warning(f'{_} will not be optimized.')

        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params_g, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)
        # optimizer d
        optim_type = train_opt['optim_d'].pop('type')
        self.optimizer_d = self.get_optimizer(optim_type, self.net_d.parameters(), **train_opt['optim_d'])
        self.optimizers.append(self.optimizer_d)

    def gray_resize_for_identity(self, out, size=128):
        out_gray = (0.2989 * out[:, 0, :, :] + 0.5870 * out[:, 1, :, :] + 0.1140 * out[:, 2, :, :])
        out_gray = out_gray.unsqueeze(1)
        out_gray = F.interpolate(out_gray, (size, size), mode='bilinear', align_corners=False)
        return out_gray

    def optimize_parameters(self, current_iter):
        # optimize net_g
        for p in self.net_d.parameters():
            p.requires_grad = False

        self.optimizer_g.zero_grad()

        large_de = False
        self.output, lq_feat, dirichletDistParam = self.net_g(self.input, w=1.0, detach_16=True)

        # if self.hq_feat_loss:
        #     # quant_feats
        #     quant_feat_gt = self.net_g.module.quantize.get_codebook_feat(self.idx_gt, shape=[self.b,16,16,256])

        l_g_total = 0
        loss_dict = OrderedDict()
        if current_iter % self.net_d_iters == 0 and current_iter > self.net_g_start_iter:
            if not large_de: # when large degradation don't need image-level loss
                # pixel loss 
                if self.cri_pix:
                    l_g_pix = self.cri_pix(self.output, self.gt)
                    l_g_total += l_g_pix
                    loss_dict['l_g_pix'] = l_g_pix

                # perceptual loss
                if self.cri_perceptual:
                    l_g_percep = self.cri_perceptual(self.output, self.gt)
                    l_g_total += l_g_percep
                    loss_dict['l_g_percep'] = l_g_percep

                if self.cri_dirichletKL:
                    l_g_dirKL = self.cri_dirichletKL(dirichletDistParam)
                    l_g_total += l_g_dirKL
                    loss_dict['l_g_dirichletKL'] = l_g_dirKL

                # gan loss
                if  current_iter > self.net_d_start_iter:
                    fake_g_pred = self.net_d(self.output)
                    l_g_gan = self.cri_gan(fake_g_pred, True, is_disc=False)
                    recon_loss = l_g_pix + l_g_percep

                    loss_dict['recon_loss'] = recon_loss
                    loss_dict['l_g_gan'] = 0.1 * l_g_gan

                    l_g_total += recon_loss
                    l_g_total += l_g_gan

            l_g_total.backward()
            
            for name, param in self.net_g.named_parameters():
                if not param.requires_grad:
                    continue

                # if param.grad is None:
                #     print(name, "f**k you!!!")
                # elif torch.all(param.grad == 0):
                #     print(name, "zero!!!!!")
            
            self.optimizer_g.step()

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

        # optimize net_d
        if not large_de:
            if current_iter > self.net_d_start_iter:
                for p in self.net_d.parameters():
                    p.requires_grad = True

                self.optimizer_d.zero_grad()
                # real
                real_d_pred = self.net_d(self.gt)
                l_d_real = self.cri_gan(real_d_pred, True, is_disc=True)
                loss_dict['l_d_real'] = l_d_real
                loss_dict['out_d_real'] = torch.mean(real_d_pred.detach())
                l_d_real.backward()
                # fake
                fake_d_pred = self.net_d(self.output.detach())
                l_d_fake = self.cri_gan(fake_d_pred, False, is_disc=True)
                loss_dict['l_d_fake'] = l_d_fake
                loss_dict['out_d_fake'] = torch.mean(fake_d_pred.detach())
                l_d_fake.backward()

                self.optimizer_d.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

    def test(self):
        with torch.no_grad():
            if hasattr(self, 'net_g_ema'):
                self.net_g_ema.eval()
                self.output, _, _ = self.net_g_ema(self.input, w=1)
            else:
                logger = get_root_logger()
                logger.warning('Do not have self.net_g_ema, use self.net_g.')
                self.net_g.eval()
                self.output, _, _ = self.net_g(self.input, w=1)
                self.net_g.train()

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
        pbar = tqdm(total=len(dataloader), unit='image')

        for idx, val_data in enumerate(dataloader):
            # img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            # img_name = val_data["key"][0].split('/')[0]
            img_name = val_data["key"][0].split('/')[-3]
            self.feed_data(val_data)
            self.test()

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']], min_max=(-1, 1))
            sr_imgs = tensor2imgs(visuals['result'], min_max=(-1, 1))
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']], min_max=(-1, 1))
                gt_imgs = tensor2imgs(visuals['gt'], min_max=(-1, 1))
                del self.gt

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}.png')
                    save_img_gif_path = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}.gif')
                    save_img_path_ori = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}_gt.png')
                    save_img_gif_path_ori = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}_gt.gif')
                else:
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["val"]["suffix"]}.png')
                        save_img_gif_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["val"]["suffix"]}.gif')
                        save_img_path_ori = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["val"]["suffix"]}_ori.png')
                        save_img_gif_path_ori = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["val"]["suffix"]}_ori.gif')
                    else:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["name"]}.png')
                        save_img_gif_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["name"]}.gif')
                        save_img_path_ori = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["name"]}_ori.png')
                        save_img_gif_path_ori = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["name"]}_ori.gif')

                imwrite(sr_img, save_img_path)
                imwrite(gt_img, save_img_path_ori)

                images_to_gif(sr_imgs, save_img_gif_path, duration = 50, loop=4)
                images_to_gif(gt_imgs, save_img_gif_path_ori, duration = 50, loop=4)
            if with_metrics:
                # calculate metrics
                for name, opt_ in self.opt['val']['metrics'].items():
                    metric_data = dict(img1=sr_img, img2=gt_img)
                    self.metric_results[name] += calculate_metric(metric_data, opt_)
            pbar.update(1)
            pbar.set_description(f'Test {img_name}')
        pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}\n'
        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{metric}', value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['gt'] = self.gt.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        if self.ema_decay > 0:
            self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter, param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_network(self.net_d, 'net_d', current_iter)
        self.save_training_state(epoch, current_iter)
