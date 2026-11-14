# Copyright (c) OpenMMLab. All rights reserved.
# Copyright 2026 The Fall Impact Monitor contributors
# SPDX-License-Identifier: Apache-2.0
#
# Adapted from https://github.com/open-mmlab/mmaction2
#   @a5a167dff2399e2d182a60332325f9c0d4663517
#   mmaction/models/backbones/resnet3d.py (Bottleneck3d, ResNet3d stem and
#     make_res_layer, as used by ResNet3dPathway)
#   mmaction/models/backbones/resnet3d_slowonly.py (ResNet3dSlowOnly)
#   mmaction/models/heads/i3d_head.py (I3DHead: pooling, dropout, linear layer)
#   configs/skeleton/posec3d/slowonly_r50_8xb16-u48-240e_ntu60-xsub-keypoint.py
# changed: plain PyTorch (nn.Conv3d, nn.BatchNorm3d, nn.ReLU in a small ConvModule
#   with the upstream attribute names instead of mmcv's); only the PoseC3D keypoint
#   configuration (17 input channels, base 32 channels, three stages of 4, 6 and 3
#   bottlenecks, inflate (0, 1, 1) in 3x1x1 style, spatial strides 2, 2, 2, temporal
#   strides 1, 1, 2, no pool2); no lateral connections, non-local blocks or
#   checkpointing; type hints. Attribute names are unchanged so MMAction2 checkpoints
#   load as they are.
"""PoseC3D's SlowOnly ResNet3D-50 backbone and I3D head (Duan et al., CVPR 2022)."""

import torch
from torch import nn


class ConvModule(nn.Module):
    """conv -> bn -> optional ReLU, with mmcv's attribute names."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int, int],
        stride: tuple[int, int, int] = (1, 1, 1),
        padding: tuple[int, int, int] = (0, 0, 0),
        activate: bool = True,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv3d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False
        )
        self.bn = nn.BatchNorm3d(out_channels)
        self.activate = nn.ReLU() if activate else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activate(self.bn(self.conv(x)))


class Bottleneck3d(nn.Module):
    expansion = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        spatial_stride: int,
        temporal_stride: int,
        inflate: bool,
        downsample: nn.Module | None,
    ) -> None:
        super().__init__()
        # 'pytorch' style: the stride is in conv2 (and the downsampling branch).
        if inflate:  # 3x1x1 inflation
            conv1_kernel, conv1_padding = (3, 1, 1), (1, 0, 0)
        else:
            conv1_kernel, conv1_padding = (1, 1, 1), (0, 0, 0)
        self.conv1 = ConvModule(inplanes, planes, conv1_kernel, padding=conv1_padding)
        self.conv2 = ConvModule(
            planes,
            planes,
            (1, 3, 3),
            stride=(temporal_stride, spatial_stride, spatial_stride),
            padding=(0, 1, 1),
        )
        self.conv3 = ConvModule(planes, planes * self.expansion, (1, 1, 1), activate=False)
        self.downsample = downsample
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.downsample is None else self.downsample(x)
        return self.relu(self.conv3(self.conv2(self.conv1(x))) + identity)


class ResNet3dSlowOnly(nn.Module):
    """The PoseC3D keypoint backbone; input (N, 17, T, H, W), output (N, 512, T', H', W')."""

    def __init__(
        self,
        in_channels: int = 17,
        base_channels: int = 32,
        stage_blocks: tuple[int, ...] = (4, 6, 3),
        inflate: tuple[int, ...] = (0, 1, 1),
        spatial_strides: tuple[int, ...] = (2, 2, 2),
        temporal_strides: tuple[int, ...] = (1, 1, 2),
    ) -> None:
        super().__init__()
        self.conv1 = ConvModule(in_channels, base_channels, (1, 7, 7), padding=(0, 3, 3))
        self.maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))
        inplanes = base_channels
        self.res_layers = []
        for i, blocks in enumerate(stage_blocks):
            planes = base_channels * 2**i
            spatial, temporal = spatial_strides[i], temporal_strides[i]
            downsample = None
            if spatial != 1 or inplanes != planes * Bottleneck3d.expansion:
                downsample = ConvModule(
                    inplanes,
                    planes * Bottleneck3d.expansion,
                    (1, 1, 1),
                    stride=(temporal, spatial, spatial),
                    activate=False,
                )
            layers = [
                Bottleneck3d(inplanes, planes, spatial, temporal, inflate[i] == 1, downsample)
            ]
            inplanes = planes * Bottleneck3d.expansion
            layers += [
                Bottleneck3d(inplanes, planes, 1, 1, inflate[i] == 1, None)
                for _ in range(1, blocks)
            ]
            name = f"layer{i + 1}"
            self.add_module(name, nn.Sequential(*layers))
            self.res_layers.append(name)
        self.feat_dim = inplanes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.maxpool(self.conv1(x))
        for name in self.res_layers:
            x = getattr(self, name)(x)
        return x


class I3DHead(nn.Module):
    def __init__(self, num_classes: int, in_channels: int, dropout_ratio: float = 0.5) -> None:
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.dropout = nn.Dropout(p=dropout_ratio)
        self.fc_cls = nn.Linear(in_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(self.avg_pool(x))
        return self.fc_cls(x.view(x.shape[0], -1))
