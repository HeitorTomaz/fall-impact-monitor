# Copyright (c) OpenMMLab. All rights reserved.
# Copyright 2026 The Fall Impact Monitor contributors
# SPDX-License-Identifier: Apache-2.0
#
# Adapted from https://github.com/open-mmlab/mmaction2
#   @a5a167dff2399e2d182a60332325f9c0d4663517
#   mmaction/models/utils/graph.py (edge2mat, normalize_digraph, Graph: coco layout,
#     spatial mode)
#   mmaction/models/utils/gcn_utils.py (unit_gcn, unit_tcn, mstcn)
#   mmaction/models/backbones/stgcn.py (STGCNBlock, STGCN)
#   mmaction/models/heads/gcn_head.py (GCNHead: pooling and linear layer)
# changed: plain PyTorch (nn.Module, nn.Sequential, nn.ModuleList, nn.BatchNorm2d,
#   nn.ReLU instead of mmengine/mmcv builders and the registry); only the COCO layout,
#   the spatial graph mode and the options ST-GCN++ uses; type hints. Attribute names
#   are unchanged so that MMAction2 checkpoints load as they are.
"""ST-GCN++ (Duan et al., PYSKL, 2022) as configured in MMAction2: COCO-17 graph in
spatial mode, ``gcn_adaptive='init'``, ``gcn_with_res=True``, ``tcn_type='mstcn'``."""

from typing import Any

import numpy as np
import torch
from torch import nn

EPS = 1e-4

COCO_INWARD = [
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 5), (12, 6), (9, 7), (7, 5),
    (10, 8), (8, 6), (5, 0), (6, 0), (1, 0), (3, 1), (2, 0), (4, 2),
]  # fmt: skip


def edge2mat(edges: list[tuple[int, int]], num_node: int) -> np.ndarray:
    A = np.zeros((num_node, num_node))
    for i, j in edges:
        A[j, i] = 1
    return A


def normalize_digraph(A: np.ndarray, dim: int = 0) -> np.ndarray:
    Dl = np.sum(A, dim)
    h, w = A.shape
    Dn = np.zeros((w, w))
    for i in range(w):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-1)
    return np.dot(A, Dn)


def coco_spatial_graph() -> np.ndarray:
    """The (3, 17, 17) adjacency of the COCO layout in spatial mode: self links, inward
    and outward edges."""
    num_node = 17
    self_link = [(i, i) for i in range(num_node)]
    outward = [(j, i) for (i, j) in COCO_INWARD]
    Iden = edge2mat(self_link, num_node)
    In = normalize_digraph(edge2mat(COCO_INWARD, num_node))
    Out = normalize_digraph(edge2mat(outward, num_node))
    return np.stack((Iden, In, Out))


class unit_gcn(nn.Module):
    """Graph convolution, ``adaptive='init'`` (a learned adjacency), ``conv_pos='pre'``."""

    def __init__(
        self, in_channels: int, out_channels: int, A: torch.Tensor, with_res: bool = True
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_subsets = A.size(0)
        self.with_res = with_res
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()
        self.A = nn.Parameter(A.clone())
        self.conv = nn.Conv2d(in_channels, out_channels * A.size(0), 1)
        if with_res:
            if in_channels != out_channels:
                self.down: Any = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1), nn.BatchNorm2d(out_channels)
                )
            else:
                self.down = lambda x: x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, c, t, v = x.shape
        res = self.down(x) if self.with_res else 0
        x = self.conv(x)
        x = x.view(n, self.num_subsets, -1, t, v)
        x = torch.einsum("nkctv,kvw->nctw", (x, self.A)).contiguous()
        return self.act(self.bn(x) + res)


class unit_tcn(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 9,
        stride: int = 1,
        dilation: int = 1,
        norm: bool = True,
        dropout: float = 0,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        pad = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            padding=(pad, 0),
            stride=(stride, 1),
            dilation=(dilation, 1),
        )
        self.bn: nn.Module = nn.BatchNorm2d(out_channels) if norm else nn.Identity()
        self.drop = nn.Dropout(dropout, inplace=True)
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.bn(self.conv(x)))


class mstcn(nn.Module):
    """Multi-scale temporal convolution."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
        ms_cfg: tuple[Any, ...] = ((3, 1), (3, 2), (3, 3), (3, 4), ("max", 3), "1x1"),
        stride: int = 1,
    ) -> None:
        super().__init__()
        self.ms_cfg = ms_cfg
        num_branches = len(ms_cfg)
        self.num_branches = num_branches
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.act = nn.ReLU()
        mid_channels = out_channels // num_branches
        rem_mid_channels = out_channels - mid_channels * (num_branches - 1)
        self.mid_channels = mid_channels
        self.rem_mid_channels = rem_mid_channels
        branches: list[nn.Module] = []
        for i, cfg in enumerate(ms_cfg):
            branch_c = rem_mid_channels if i == 0 else mid_channels
            if cfg == "1x1":
                branches.append(nn.Conv2d(in_channels, branch_c, kernel_size=1, stride=(stride, 1)))
                continue
            if cfg[0] == "max":
                branches.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, branch_c, kernel_size=1),
                        nn.BatchNorm2d(branch_c),
                        self.act,
                        nn.MaxPool2d(kernel_size=(cfg[1], 1), stride=(stride, 1), padding=(1, 0)),
                    )
                )
                continue
            branches.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, branch_c, kernel_size=1),
                    nn.BatchNorm2d(branch_c),
                    self.act,
                    unit_tcn(
                        branch_c,
                        branch_c,
                        kernel_size=cfg[0],
                        stride=stride,
                        dilation=cfg[1],
                        norm=False,
                    ),
                )
            )
        self.branches = nn.ModuleList(branches)
        tin_channels = mid_channels * (num_branches - 1) + rem_mid_channels
        self.transform = nn.Sequential(
            nn.BatchNorm2d(tin_channels),
            self.act,
            nn.Conv2d(tin_channels, out_channels, kernel_size=1),
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.drop = nn.Dropout(dropout, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = torch.cat([branch(x) for branch in self.branches], dim=1)
        return self.drop(self.bn(self.transform(feat)))


class STGCNBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        A: torch.Tensor,
        stride: int = 1,
        residual: bool = True,
        tcn_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.gcn = unit_gcn(in_channels, out_channels, A, with_res=True)
        self.tcn = mstcn(out_channels, out_channels, dropout=tcn_dropout, stride=stride)
        self.relu = nn.ReLU()
        if not residual:
            self.residual: Any = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.residual(x)
        x = self.tcn(self.gcn(x)) + res
        return self.relu(x)


class STGCN(nn.Module):
    """Backbone: input (N, M persons, T frames, V=17 joints, C channels)."""

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        ch_ratio: int = 2,
        num_stages: int = 10,
        inflate_stages: tuple[int, ...] = (5, 8),
        down_stages: tuple[int, ...] = (5, 8),
    ) -> None:
        super().__init__()
        A = torch.tensor(coco_spatial_graph(), dtype=torch.float32, requires_grad=False)
        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.ch_ratio = ch_ratio
        self.inflate_stages = inflate_stages
        self.down_stages = down_stages
        modules = []
        if self.in_channels != self.base_channels:
            modules = [STGCNBlock(in_channels, base_channels, A.clone(), 1, residual=False)]
        inflate_times = 0
        for i in range(2, num_stages + 1):
            stride = 1 + (i in down_stages)
            in_channels = base_channels
            if i in inflate_stages:
                inflate_times += 1
            out_channels = int(self.base_channels * self.ch_ratio**inflate_times + EPS)
            base_channels = out_channels
            modules.append(STGCNBlock(in_channels, out_channels, A.clone(), stride))
        if self.in_channels == self.base_channels:
            num_stages -= 1
        self.num_stages = num_stages
        self.gcn = nn.ModuleList(modules)
        self.out_channels = base_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, M, T, V, C = x.size()
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = self.data_bn(x.view(N * M, V * C, T))
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)
        for i in range(self.num_stages):
            x = self.gcn[i](x)
        return x.reshape((N, M, *x.shape[1:]))


class GCNHead(nn.Module):
    """Average over persons, frames and joints, then a linear layer."""

    def __init__(self, num_classes: int, in_channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.dropout = nn.Dropout(p=dropout) if dropout else None
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, M, C, T, V = x.shape
        x = self.pool(x.view(N * M, C, T, V)).view(N, M, C).mean(dim=1)
        if self.dropout is not None:
            x = self.dropout(x)
        return self.fc(x)
