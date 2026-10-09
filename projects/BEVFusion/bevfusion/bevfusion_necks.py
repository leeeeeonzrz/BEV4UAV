# modify from https://github.com/mit-han-lab/bevfusion
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule

from mmdet3d.registry import MODELS


@MODELS.register_module()
class GeneralizedLSSFPN(BaseModule):

    def __init__(
            self,
            in_channels,
            out_channels,
            num_outs,
            start_level=0,
            end_level=-1,
            no_norm_on_lateral=False,
            conv_cfg=None,
            norm_cfg=dict(type='BN2d'),
            act_cfg=dict(type='ReLU'),
            upsample_cfg=dict(mode='bilinear', align_corners=True),
    ) -> None:
        super().__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()

        if end_level == -1:
            self.backbone_end_level = self.num_ins - 1
            # assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i] +
                (in_channels[i + 1] if i == self.backbone_end_level -
                 1 else out_channels),
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=act_cfg,
                inplace=False,
            )
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False,
            )

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

    def forward(self, inputs):
        """Forward function."""
        # upsample -> cat -> conv1x1 -> conv3x3
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [inputs[i + self.start_level] for i in range(len(inputs))]

        # build top-down path
        used_backbone_levels = len(laterals) - 1
        for i in range(used_backbone_levels - 1, -1, -1):
            x = F.interpolate(
                laterals[i + 1],
                size=laterals[i].shape[2:],
                **self.upsample_cfg,
            )
            laterals[i] = torch.cat([laterals[i], x], dim=1)
            laterals[i] = self.lateral_convs[i](laterals[i])
            laterals[i] = self.fpn_convs[i](laterals[i])

        # build outputs
        outs = [laterals[i] for i in range(used_backbone_levels)]
        return tuple(outs)



# @MODELS.register_module()
# class MergedLSSFPN(BaseModule):
#     def __init__(
#         self,
#         in_channels,
#         out_channels,
#         num_outs=1,  # 固定输出单个融合特征
#         start_level=0,
#         end_level=-1,
#         conv_cfg=None,
#         norm_cfg=dict(type="BN2d"),
#         act_cfg=dict(type="ReLU"),
#         upsample_cfg=dict(mode="bilinear", align_corners=True),
#     ) -> None:
#         super().__init__()
#         assert isinstance(in_channels, list)
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.num_ins = len(in_channels)
#         self.upsample_cfg = upsample_cfg.copy()

#         # 确定特征处理范围
#         if end_level == -1:
#             self.backbone_end_level = self.num_ins - 1
#         else:
#             self.backbone_end_level = end_level
#         self.start_level = start_level

#         # 为每个输入层创建通道调整卷积 (注意：此处的卷积用于调整通道，输入是原始通道)
#         self.lateral_convs = nn.ModuleList()
#         for i in range(self.start_level, self.backbone_end_level + 1):
#             l_conv = ConvModule(
#                 in_channels[i], # 输入通道为 backbone 原始输出通道
#                 out_channels,    # 输出通道统一为 out_channels
#                 1,
#                 conv_cfg=conv_cfg,
#                 norm_cfg=norm_cfg,
#                 act_cfg=act_cfg,
#                 inplace=False,
#             )
#             self.lateral_convs.append(l_conv)

#         # 最终融合卷积 (输入通道数为所有特征图数量 * out_channels)
#         num_feature_maps = self.backbone_end_level - self.start_level + 1
#         self.fusion_conv = ConvModule(
#             out_channels * num_feature_maps,
#             out_channels,
#             3,
#             padding=1,
#             conv_cfg=conv_cfg,
#             norm_cfg=norm_cfg,
#             act_cfg=act_cfg,
#             inplace=False,
#         )

#     def forward(self, inputs):
#         # 确定基准分辨率（通常是最底层或最大空间尺寸的特征图）
#         base_size = inputs[self.start_level].shape[2:] # [H, W]

#         # 1. 先对每个输入特征进行上采样（如果需要）到基准分辨率
#         upsampled_features = []
#         for i in range(self.start_level, self.backbone_end_level + 1):
#             x = inputs[i]
#             # 如果当前特征图尺寸与基准尺寸不同，则进行上采样
#             if x.shape[2:] != base_size:
#                 x = F.interpolate(
#                     x,
#                     size=base_size,
#                     **self.upsample_cfg,
#                 )
#             upsampled_features.append(x)

#         # 2. 对上采样后的特征进行通道调整
#         laterals = []
#         for i, feat in enumerate(upsampled_features):
#             # 使用对应的卷积模块调整通道数 (i 对应 lateral_convs 中的索引)
#             x = self.lateral_convs[i](feat) # 输入是上采样后的特征，通道数为 in_channels[i]
#             laterals.append(x)

#         # 3. 在通道维度上拼接所有经过通道调整后的特征
#         fused = torch.cat(laterals, dim=1) # [batch_size, out_channels * num_levels, H, W]

#         # 4. 使用融合卷积进一步融合特征并可能调整最终通道数
#         out = self.fusion_conv(fused) # [batch_size, out_channels, H, W]

#         return (out,)  # 返回元组保持接口兼容