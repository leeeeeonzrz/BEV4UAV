import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
# from mmengine.runner import BaseModule, auto_fp16
from mmengine.model import BaseModule
# from mmengine.runner import auto_fp16
# from mmdet.models.builder import NECKS
from mmdet3d.registry import MODELS


@MODELS.register_module()
class MergedLSSFPN1(BaseModule):
    def __init__(
        self,
        in_channels,
        out_channels,
        num_outs=1,  # 固定输出单个融合特征
        start_level=0,
        end_level=-1,
        conv_cfg=None,
        norm_cfg=dict(type="BN2d"),
        act_cfg=dict(type="ReLU"),
        upsample_cfg=dict(mode="bilinear", align_corners=True),
    ) -> None:
        super().__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.upsample_cfg = upsample_cfg.copy()

        # 确定特征处理范围
        if end_level == -1:
            self.backbone_end_level = self.num_ins - 1
        else:
            self.backbone_end_level = end_level
        self.start_level = start_level

        # 为每个输入层创建通道调整卷积
        self.lateral_convs = nn.ModuleList()
        for i in range(self.start_level, self.backbone_end_level + 1):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False,
            )
            self.lateral_convs.append(l_conv)

        # 最终融合卷积
        self.fusion_conv = ConvModule(
            out_channels * (self.backbone_end_level - self.start_level + 1),
            out_channels,
            3,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            inplace=False,
        )

    # @auto_fp16()
    def forward(self, inputs):
        # 确定基准分辨率（最底层特征的分辨率）
        base_size = inputs[self.start_level].shape[2:]
        
        # 处理每个输入特征
        laterals = []
        for i in range(self.start_level, self.backbone_end_level + 1):
            x = inputs[i]
            # 通道调整
            x = self.lateral_convs[i - self.start_level](x)
            # 上采样到基准分辨率
            # 上采样到基准分辨率（修正后）
            if x.shape[2:] != base_size:
                x = F.interpolate(
                    x, 
                    size=base_size,
                    **self.upsample_cfg,   # 无隐藏字符
                )
            laterals.append(x)
        
        # 拼接所有特征
        fused = torch.cat(laterals, dim=1)
        
        # 最终融合
        out = self.fusion_conv(fused)
        return (out,)  # 返回元组保持接口兼容