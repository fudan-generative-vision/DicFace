import os
import random
from pathlib import Path

from PIL import Image
import cv2
import ffmpeg
import io
import av
import numpy as np
import torch
from torchvision.transforms.functional import normalize
from basicsr.data.degradations import (random_add_gaussian_noise,
                                       random_mixed_kernels)
from basicsr.data.data_util import paths_from_folder, brush_stroke_mask, brush_stroke_mask_video, random_ff_mask
from basicsr.data.transforms import augment
from basicsr.utils import FileClient, get_root_logger, img2tensor, imfrombytes, scandir
from basicsr.utils.registry import DATASET_REGISTRY
from facelib.utils.face_restoration_helper import FaceAligner
from torch.utils import data as data

@DATASET_REGISTRY.register()
class InpaintingDataset(data.Dataset):
    def __init__(self, opt):
        super(InpaintingDataset, self).__init__()
        self.opt = opt
        self.gt_root = Path(opt['dataroot_gt'])

        self.num_frame = opt['video_length'] # 5
        self.scale = opt['scale'] # [1, 4]
        self.need_align = opt.get('need_align', False) # False
        self.normalize = opt.get('normalize', False) # True

        self.keys = []
        with open(opt['global_meta_info_file'], 'r') as fin:
            for line in fin:
                real_clip_path = '/'.join(line.split('/')[:-1])
                clip_length = int(line.split('/')[-1])
                self.keys.extend([f'{real_clip_path}/{clip_length:08d}/{0:08d}'])

        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.is_lmdb = False
        if self.io_backend_opt['type'] == 'lmdb':
            self.is_lmdb = True
            self.io_backend_opt['db_paths'] = [self.gt_root]
            self.io_backend_opt['client_keys'] = ['gt']

        # temporal augmentation configs
        self.interval_list = opt['interval_list'] # [1]
        self.random_reverse = opt['random_reverse']
        interval_str = ','.join(str(x) for x in opt['interval_list']) # '1'
        logger = get_root_logger()
        logger.info(f'Temporal augmentation interval list: [{interval_str}]; '
                    f'random reverse is {self.random_reverse}.')

        # degradations
        # blur
        self.blur_kernel_size = opt['blur_kernel_size'] # 21
        self.kernel_list = opt['kernel_list']           # ['iso', 'aniso']
        self.kernel_prob = opt['kernel_prob']           # [0.5, 0.5]  
        self.blur_x_sigma = opt['blur_x_sigma']         # [0.2, 3]
        self.blur_y_sigma = opt['blur_y_sigma']         # [0.2, 3]
        # noise
        self.noise_range = opt['noise_range']           # [0, 25] 
        # resize
        self.resize_prob = opt['resize_prob']           # [0.25, 0.25, 0.5]
        # crf
        self.crf_range = opt['crf_range']               # [10, 30]
        # codec
        self.vcodec = opt['vcodec']                     # ['libx264']
        self.vcodec_prob = opt['vcodec_prob']           # [1]

        logger.info(f'Blur: blur_kernel_size {self.blur_kernel_size}, '
                    f'x_sigma: [{", ".join(map(str, self.blur_x_sigma))}], '
                    f'y_sigma: [{", ".join(map(str, self.blur_y_sigma))}], ')
        logger.info(f'Noise: [{", ".join(map(str, self.noise_range))}]')
        logger.info(f'CRF compression: [{", ".join(map(str, self.crf_range))}]')
        logger.info(f'Codec: [{", ".join(map(str, self.vcodec))}]')

        if self.need_align:
            self.dataroot_meta_info = opt['dataroot_meta_info']
            self.face_aligner = FaceAligner(
                upscale_factor=1,
                face_size=512,
                crop_ratio=(1, 1),
                det_model='retinaface_resnet50',
                save_ext='png',
                use_parse=True)

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        key = self.keys[index]
        real_clip_path = '/'.join(key.split('/')[:-2])
        clip_length = int(key.split('/')[-2])
        frame_idx = int(key.split('/')[-1])
        clip_name = real_clip_path.split('/')[-1]

        if os.path.exists(os.path.join(self.gt_root, "train", clip_name)):
            paths = sorted(list(scandir(os.path.join(self.gt_root, "train", clip_name))))
        elif os.path.exists(os.path.join(self.gt_root, "test", clip_name)):
            paths = sorted(list(scandir(os.path.join(self.gt_root, "test", clip_name))))
        else:
            paths = sorted(list(scandir(os.path.join(self.gt_root, clip_name))))

        # determine the neighboring frames
        interval = random.choice(self.interval_list)

        # exceed the length, re-select a new clip
        while (clip_length - self.num_frame * interval) < 0:
            interval = random.choice(self.interval_list)

        # ensure not exceeding the borders
        start_frame_idx = frame_idx - self.num_frame // 2 * interval
        end_frame_idx = frame_idx + (self.num_frame + 1) // 2 * interval

        while (start_frame_idx < 0) or (end_frame_idx > clip_length):
            frame_idx = random.randint(self.num_frame // 2 * interval,
                                       clip_length - self.num_frame // 2 * interval)
            start_frame_idx = frame_idx - self.num_frame // 2 * interval
            end_frame_idx = frame_idx + (self.num_frame + 1) // 2 * interval
        neighbor_list = list(range(start_frame_idx, end_frame_idx, interval))

        # random reverse
        if self.random_reverse and random.random() < 0.5:
            neighbor_list.reverse()

        assert len(neighbor_list) == self.num_frame, (
            f'Wrong length of neighbor list: {len(neighbor_list)}')

        # get the neighboring GT frames
        img_gts = []

        need_align = False
        if self.need_align:
            clip_info_path = os.path.join(self.dataroot_meta_info, f'{clip_name}.txt')
            if os.path.exists(clip_info_path):
                need_align = True
                clip_info = []
                with open(clip_info_path, 'r', encoding='utf-8') as fin:
                    for line in fin:
                        line = line.strip()
                        clip_info.append(line)

        for neighbor in neighbor_list:
            img_gt_path = os.path.join(self.gt_root, clip_name, paths[neighbor])
            if not os.path.exists(img_gt_path):
                img_gt_path = os.path.join(self.gt_root, "train", clip_name, paths[neighbor])
            if not os.path.exists(img_gt_path):
                img_gt_path = os.path.join(self.gt_root, "test", clip_name, paths[neighbor])

            img_gt = np.asarray(Image.open(img_gt_path))[:, :, ::-1] / 255.0
            img_gts.append(img_gt)

        # augmentation - flip, rotate
        img_gts = augment(img_gts, self.opt['use_flip'], self.opt['use_rot']) # False, False

        # ------------- generate inpaint frames --------------#
        img_lqs = img_gts
        img_lqs = [Image.fromarray((_ * 255).astype('uint8')) for _ in img_lqs]
        img_lqs = brush_stroke_mask_video(img_lqs)
        img_lqs = [np.array(_) / 255. for _ in img_lqs]

        # ------------ Align -------------#
        if need_align:
            align_lqs, align_gts = [], []
            for frame_idx, (img_lq, img_gt) in enumerate(zip(img_lqs, img_gts)):
                landmarks_str = clip_info[start_frame_idx + frame_idx].split(' ')
                landmarks = np.array([float(x) for x in landmarks_str]).reshape(5, 2)
                self.face_aligner.clean_all()

                # align and warp each face
                img_lq, img_gt = self.face_aligner.align_pair_face(img_lq, img_gt, landmarks)
                align_lqs.append(img_lq)
                align_gts.append(img_gt)
            img_lqs, img_gts = align_lqs, align_gts

        img_gts = img2tensor(img_gts)
        img_lqs = img2tensor(img_lqs)
        img_gts = torch.stack(img_gts, dim=0)
        img_lqs = torch.stack(img_lqs, dim=0)

        if self.normalize:
            normalize(img_lqs, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], inplace=True)
            normalize(img_gts, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], inplace=True)

        return {'in': img_lqs, 'gt': img_gts, 'key': key}

    def __len__(self):
        return len(self.keys)
