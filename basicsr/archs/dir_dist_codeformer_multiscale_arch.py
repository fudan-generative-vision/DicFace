import math
import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from typing import Optional, List

from basicsr.archs.vqgan_arch import *
from basicsr.utils import get_root_logger
from basicsr.utils.registry import ARCH_REGISTRY

import torch.distributions as dist

from einops import rearrange

def calc_mean_std(feat, eps=1e-5):
    """Calculate mean and std for adaptive_instance_normalization.

    Args:
        feat (Tensor): 4D tensor.
        eps (float): A small value added to the variance to avoid
            divide-by-zero. Default: 1e-5.
    """
    size = feat.size()
    assert len(size) == 4, 'The input feature should be 4D tensor.'
    b, c = size[:2]
    feat_var = feat.view(b, c, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(b, c, 1, 1)
    feat_mean = feat.view(b, c, -1).mean(dim=2).view(b, c, 1, 1)
    return feat_mean, feat_std


def adaptive_instance_normalization(content_feat, style_feat):
    """Adaptive instance normalization.

    Adjust the reference features to have the similar color and illuminations
    as those in the degradate features.

    Args:
        content_feat (Tensor): The reference feature.
        style_feat (Tensor): The degradate features.
    """
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)
    normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x, mask=None):
        if mask is None:
            mask = torch.zeros((x.size(0), x.size(2), x.size(3)), device=x.device, dtype=torch.bool)
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")


class TransformerSALayer(nn.Module):
    def __init__(self, embed_dim, nhead=8, dim_mlp=2048, dropout=0.0, activation="gelu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, nhead, dropout=dropout)
        # Implementation of Feedforward model - MLP
        self.linear1 = nn.Linear(embed_dim, dim_mlp)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_mlp, embed_dim)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(
        self,
        tgt,
        tgt_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):

        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q,
                              k,
                              value=tgt2,
                              attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)

        # ffn
        tgt2 = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout2(tgt2)
        return tgt
 
 
class TransformerSALayerTemporal(nn.Module):
    def __init__(self, embed_dim, nhead=8, dim_mlp=2048, dropout=0.0, activation="gelu"):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(embed_dim, nhead, dropout=dropout)
        # Implementation of Feedforward model - MLP
        self.linear1 = nn.Linear(embed_dim, dim_mlp)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_mlp, embed_dim)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self,
                tgt,
                frame_length=10,
                batch_size=1,
                tgt_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):

        tgt = rearrange(tgt, "d (b t) c -> t (b d) c", t=frame_length)

        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q,
                              k,
                              value=tgt2,
                              attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)

        # ffn
        tgt2 = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout2(tgt2)
        # reshape
        tgt = rearrange(tgt, "t (b d) c -> d (b t) c", b=batch_size)

        return tgt


class Fuse_sft_block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.encode_enc = ResBlock(2*in_ch, out_ch)

        self.scale = nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                    nn.LeakyReLU(0.2, True),
                    nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1))

        self.shift = nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                    nn.LeakyReLU(0.2, True),
                    nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1))

    def forward(self, enc_feat, dec_feat, w=1):
        enc_feat = self.encode_enc(torch.cat([enc_feat, dec_feat], dim=1))
        scale = self.scale(enc_feat)
        shift = self.shift(enc_feat)
        residual = w * (dec_feat * scale + shift)
        out = dec_feat + residual
        return out

class ExpModule(nn.Module):
    def forward(self, x):
        return torch.exp(x)


class MultiScaleFuse(nn.Module):
    def __init__(self):
        super(MultiScaleFuse, self).__init__()
        self.s64_conv = nn.Conv2d(in_channels=256*16, out_channels=256, kernel_size=1)
        self.s32_conv = nn.Conv2d(in_channels=256*4, out_channels=256, kernel_size=1)
        self.s16_conv = nn.Conv2d(in_channels=256*1, out_channels=256, kernel_size=1)
        self.out = nn.Conv2d(in_channels=256*3, out_channels=256, kernel_size=3, stride=1, padding=1)

    def forward(self, s64, s32, s16):

        feat_64 = rearrange(s64, "bt c (h h1) (w w1) -> bt (c h1 w1) h w", h1=4, w1=4) 
        feat_64 = self.s64_conv(feat_64)
        feat_32 = rearrange(s32, "bt c (h h1) (w w1) -> bt (c h1 w1) h w", h1=2, w1=2) 
        feat_32 = self.s32_conv(feat_32)
        feat_16 = self.s16_conv(s16)

        out = self.out(torch.concat([feat_64, feat_32, feat_16], dim=1))
        return out

@ARCH_REGISTRY.register()
class TemporalCodeFormerDirDistMultiScale(VQAutoEncoder):
    def __init__(self,
                 dim_embed=512,
                 n_head=8,
                 n_layers=9, 
                 codebook_size=1024,
                 latent_size=256,
                 connect_list=['32', '64', '128', '256'],
                 fix_modules=['quantize','generator'],
                 vqgan_path=None,
                 frame_length=10,
                 new_codebook_size=None):
        super(TemporalCodeFormerDirDistMultiScale, self).__init__(512, 64, [1, 2, 2, 4, 4, 8], 'nearest', 2, [16], codebook_size)

        if vqgan_path is not None:
            self.load_state_dict(
                torch.load(vqgan_path, map_location='cpu')['params_ema'])

        self.frame_length = frame_length

        self.connect_list = connect_list
        self.n_layers = n_layers
        self.dim_embed = dim_embed
        self.dim_mlp = dim_embed * 2

        self.position_emb = nn.Parameter(torch.zeros(latent_size, self.dim_embed))
        self.position_emb_temporal = nn.Parameter(torch.zeros(self.frame_length, self.dim_embed))
        self.feat_emb = nn.Linear(256, self.dim_embed)

        self.codebook_size = codebook_size
        self.new_codebook_size = None
        if new_codebook_size is not None:
            self.new_codebook_size = new_codebook_size
            self.codebook_size += new_codebook_size
            self.new_codebook = nn.Parameter(torch.normal(mean=0, std=0.75, size=(new_codebook_size, 256)))
            self.new_codebook.requires_grad = True

        self.multiscale = MultiScaleFuse()

        # transformer in Space
        self.ft_layers = nn.Sequential(*[TransformerSALayer(embed_dim=dim_embed,
                                                            nhead=n_head,
                                                            dim_mlp=self.dim_mlp,
                                                            dropout=0.1)
                                    for _ in range(self.n_layers)])
        # transformer in Temporal
        self.dir_dist_layers = nn.Sequential(*[TransformerSALayerTemporal(embed_dim=dim_embed,
                                                                          nhead=n_head,
                                                                          dim_mlp=self.dim_mlp,
                                                                          dropout=0.1)
                                    for _ in range(self.n_layers)])

        # logits_predict head
        self.idx_pred_layer = nn.Sequential(
            nn.LayerNorm(dim_embed),
            nn.Linear(dim_embed, self.codebook_size, bias=False),
        )

        self.channels = {
            '16': 512,
            '32': 256,
            '64': 256,
            '128': 128,
            '256': 128,
            '512': 64,
        }


        self.fuse_encoder_block = {'512':2, '256':5, '128':8, '64':11, '32':14, '16':18}
        self.fuse_generator_block = {'16':6, '32': 9, '64':12, '128':15, '256':18, '512':21}

        # fuse_convs_dict
        self.fuse_convs_dict = nn.ModuleDict()
        for f_size in self.connect_list:
            in_ch = self.channels[f_size]
            self.fuse_convs_dict[f_size] = Fuse_sft_block(in_ch, in_ch)

        self.softplus_layer = nn.Softplus()
        self.position_emb.requires_grad = False
        print("Module: position_emb_spatial Frozen!")

        if fix_modules is not None:
            print(fix_modules, "frozen!")
            for module in fix_modules:
                for param_name, param in getattr(self, module).named_parameters():
                    if "conv3d" in param_name:
                        param.requires_grad = True
                    else:
                        # print(f"Module: {module}, Parameter name: {param_name} Frozen!")
                        param.requires_grad = False

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x, w=0, detach_16=True, code_only=False, adain=False):
        # ################### Encoder #####################
        enc_feat_dict = {}
        out_list = [self.fuse_encoder_block[f_size] for f_size in self.connect_list]
        for i, block in enumerate(self.encoder.blocks):
            x = block(x) 
            if i in out_list:
                enc_feat_dict[str(x.shape[-1])] = x.clone()

        lq_feat = self.multiscale(enc_feat_dict['64'], enc_feat_dict['32'], x)

        bt, c, h, width = lq_feat.shape
        b = bt // self.frame_length
        t = self.frame_length
        # ################# Spatial & Temporal Transformers ###################
        spatial_pos_emb = self.position_emb.unsqueeze(1).repeat(1, bt, 1)
        temporal_pos_emb = self.position_emb_temporal.unsqueeze(1).repeat(1, b*h*width, 1)
        feat_emb = self.feat_emb(lq_feat.flatten(2).permute(2, 0, 1))
        query_emb = feat_emb

        for layer_space, layer_temporal in zip(self.ft_layers, self.dir_dist_layers):
            query_emb = layer_space(query_emb, query_pos=spatial_pos_emb)
            query_emb = layer_temporal(query_emb, query_pos=temporal_pos_emb, frame_length=t, batch_size=b)

        alpha = self.idx_pred_layer(query_emb)
        alpha = alpha.permute(1, 0, 2)
        alpha = self.softplus_layer(alpha) + 1e-2

        dirichlet_dist = dist.Dirichlet(alpha)
        parameters = dirichlet_dist.rsample()

        parameters_reshaped = parameters.reshape(-1, self.codebook_size)

        if self.new_codebook_size is not None:
            quant_feat = torch.matmul(parameters_reshaped[:, :-self.new_codebook_size], self.quantize.embedding.weight) + \
                 torch.matmul(parameters_reshaped[:, -self.new_codebook_size:], self.new_codebook)
        else:
            quant_feat = torch.matmul(parameters_reshaped, self.quantize.embedding.weight) 

        quant_feat = rearrange(quant_feat, "(b t h w) c -> (b t) c h w", b=b, t=t, h=h, w=width)


        if adain:
            quant_feat = adaptive_instance_normalization(quant_feat, lq_feat)

        # ################## Generator ####################
        x = quant_feat
        fuse_list = [self.fuse_generator_block[f_size] for f_size in self.connect_list]

        for i, block in enumerate(self.generator.blocks):
            x = block(x) 
            if i in fuse_list:
                f_size = str(x.shape[-1])
                if w > 0:
                    x = self.fuse_convs_dict[f_size](enc_feat_dict[f_size].detach(), x, w)
        out = x
        return out, lq_feat, alpha + 1e-6
