import math
import lpips
import torch
from torch import autograd as autograd
from torch import nn as nn
from torch.nn import functional as F

from basicsr.archs.vgg_arch import VGGFeatureExtractor
from basicsr.utils.registry import LOSS_REGISTRY
# from .loss_util import weighted_loss
from basicsr.losses.loss_util import weighted_loss

_reduction_modes = ['none', 'mean', 'sum']

LPIPS_MODEL_PATH = "/sykj_002/workspaces/HanlinShang/FVR_Project/FVR_project/weights/vgg/vgg16-397923af.pth"

class LPIPSLoss(nn.Module):
    def __init__(self, 
            loss_weight=1.0, 
            use_input_norm=True,
            range_norm=False,):
        super(LPIPSLoss, self).__init__()
        self.perceptual = lpips.LPIPS(net="vgg", spatial=False, pnet_rand=True, model_path=LPIPS_MODEL_PATH).eval()
        self.loss_weight = loss_weight
        self.use_input_norm = use_input_norm
        self.range_norm = range_norm

        if self.use_input_norm:
            # the mean is for image with range [0, 1]
            self.register_buffer('mean', torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            # the std is for image with range [0, 1]
            self.register_buffer('std', torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, pred, target):
        if self.range_norm:
            pred   = (pred + 1) / 2
            target = (target + 1) / 2
        if self.use_input_norm:
            pred   = (pred - self.mean) / self.std
            target = (target - self.mean) / self.std
        lpips_loss = self.perceptual(target.contiguous(), pred.contiguous())
        return self.loss_weight * lpips_loss.mean()


if __name__ == '__main__':
    torch.manual_seed(42)
    
    # 固定输入张量（示例：全0图像和全0.5图像，差异固定）
    device = torch.device("cpu")  # 或"cuda"
    pred = torch.full((2, 3, 64, 64), 0.0, device=device)  # 全0图像
    target = torch.full((2, 3, 64, 64), 0.5, device=device)  # 全0.5图像
    
    loss_fn = LPIPSLoss().to(device)
    
    # 前向传播
    with torch.no_grad():
        loss = loss_fn(pred, target)
    
    print(f"固定输入的LPIPS损失值: {loss.item()}")
    print("输入形状:", pred.shape, target.shape)
    print("模型设备:", next(loss_fn.parameters()).device)