from typing import List
import math
import torch
from torch import nn
# from mmdet3d.models.builder import FUSERS
from mmdet3d.registry import MODELS

# __all__ = ["EMAFuser"]


def make_group_norm(num_channels: int, max_groups: int = 32) -> nn.GroupNorm:
    num_groups = math.gcd(num_channels, max_groups)
    if num_groups == 0:
        num_groups = 1
    return nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)

class EMA(nn.Module):
    def __init__(self, channels, c2=None, factor=32):
        super(EMA, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)

@MODELS.register_module()
class EMAFuser(nn.Module):
    def __init__(self, 
                 in_channels: List[int],
                 out_channels: int,
                 ema_factor:int = 32) -> None:
        super().__init__()
        self.conv_fuser = nn.Sequential(
            nn.Conv2d(sum(in_channels), out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )
        self.ema = EMA(channels=out_channels, factor=ema_factor)

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        x = torch.cat(inputs, dim=1)
        x = self.conv_fuser(x)
        x = self.ema(x)
        return x


@MODELS.register_module()
class LidarResidualEMAFuser(nn.Module):
    """Fuse camera features as a residual on top of pretrained lidar BEV.

    The learnable gate starts from zero, so the fused feature is exactly the
    lidar feature at initialization. This avoids destroying the strong
    lidar-only checkpoint before the camera branch has learned anything useful.
    """

    def __init__(self,
                 in_channels: List[int],
                 out_channels: int,
                 ema_factor: int = 32,
                 gate_init: float = 0.0,
                 max_residual_scale: float = 1.0) -> None:
        super().__init__()
        assert len(in_channels) == 2, (
            'LidarResidualEMAFuser expects [camera_bev_channels, '
            'lidar_bev_channels].')
        cam_channels, lidar_channels = in_channels
        assert lidar_channels == out_channels, (
            'out_channels must match lidar channels so the residual path can '
            'preserve the pretrained lidar feature at initialization.')

        self.cam_encoder = nn.Sequential(
            nn.Conv2d(cam_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )
        self.ema = EMA(channels=out_channels, factor=ema_factor)
        self.camera_scale = nn.Parameter(torch.tensor(float(gate_init)))
        self.max_residual_scale = float(max_residual_scale)

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        cam_feature, lidar_feature = inputs
        cam_residual = self.cam_encoder(cam_feature)
        cam_residual = self.ema(cam_residual)
        residual_scale = self.max_residual_scale * torch.tanh(
            self.camera_scale)
        return lidar_feature + residual_scale * cam_residual


@MODELS.register_module()
class LidarResidualGNEMAFuser(LidarResidualEMAFuser):
    """GroupNorm variant for stable batch_size=1 camera residual learning."""

    def __init__(self,
                 in_channels: List[int],
                 out_channels: int,
                 ema_factor: int = 32,
                 gate_init: float = 0.0,
                 max_residual_scale: float = 1.0) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            ema_factor=ema_factor,
            gate_init=gate_init,
            max_residual_scale=max_residual_scale)
        cam_channels, _ = in_channels
        self.cam_encoder = nn.Sequential(
            nn.Conv2d(cam_channels, out_channels, 3, padding=1, bias=False),
            make_group_norm(out_channels),
            nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            make_group_norm(out_channels),
            nn.ReLU(True),
        )
