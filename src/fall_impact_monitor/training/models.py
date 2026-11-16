"""Temporal classifier architectures (#15). Input ``(batch, length, features)`` pose
windows (``analysis.pose_window``); output one fall logit per window."""

import torch
from torch import nn

from fall_impact_monitor._vendor.flash.hypermamba import HyperMamba
from fall_impact_monitor._vendor.mmaction2.posec3d import I3DHead, ResNet3dSlowOnly
from fall_impact_monitor._vendor.mmaction2.stgcn import STGCN, GCNHead
from fall_impact_monitor.analysis.pose_window import KEYPOINTS


class _CausalBlock(nn.Module):
    """Dilated causal convolution with a residual connection: a row sees only itself and
    earlier rows."""

    def __init__(self, channels: int, kernel: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.pad = (kernel - 1) * dilation
        self.conv = nn.Conv1d(channels, channels, kernel, dilation=dilation)
        self.norm = nn.BatchNorm1d(channels)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(nn.functional.pad(x, (self.pad, 0)))
        return x + self.drop(torch.relu(self.norm(y)))


class SmallTCN(nn.Module):
    """The cost and quality baseline: a few dilated causal convolutions over the window,
    read at its last row."""

    def __init__(
        self,
        features: int,
        channels: int = 32,
        dilations: tuple[int, ...] = (1, 2, 4, 8),
        kernel: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.inp = nn.Conv1d(features, channels, 1)
        self.blocks = nn.Sequential(
            *(_CausalBlock(channels, kernel, d, dropout) for d in dilations)
        )
        self.head = nn.Linear(channels, 1)

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        x = self.blocks(self.inp(windows.transpose(1, 2)))
        return self.head(x[:, :, -1]).squeeze(-1)


class TCNTE(nn.Module):
    """TCN + Transformer encoder, reimplemented from its published description (Yu et
    al., Pervasive and Mobile Computing 107, 2025; no code was published and the full
    paper was not available here: layer sizes, depth and the readout are this project's
    choices). Causal convolutions extract short-range motion; self-attention relates the
    window's rows. Padding rows (``observed`` = 0) are zeroed and masked."""

    def __init__(
        self,
        features: int,
        length: int,
        channels: int = 64,
        heads: int = 4,
        layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.inp = nn.Conv1d(features, channels, 1)
        self.tcn = nn.Sequential(*(_CausalBlock(channels, 3, d, dropout) for d in (1, 2)))
        self.position = nn.Parameter(torch.zeros(1, length, channels))
        layer = nn.TransformerEncoderLayer(channels, heads, 2 * channels, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.head = nn.Linear(channels, 1)

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        observed = windows[:, :, -1:]
        x = self.tcn(self.inp((windows * observed).transpose(1, 2))).transpose(1, 2)
        x = self.encoder(x + self.position, src_key_padding_mask=observed[:, :, 0] < 0.5)
        return self.head(x[:, -1]).squeeze(-1)


class STGCNPP(nn.Module):
    """ST-GCN++ (MMAction2 configuration, ported in ``_vendor/mmaction2``) on the
    window's keypoints (x, y, confidence) as one person; ``backbone`` and ``cls_head``
    keep MMAction2's names so its published checkpoints load (``training.pretrained``).
    The time gaps are not an input: the published weights expect three channels."""

    def __init__(self) -> None:
        super().__init__()
        self.backbone = STGCN(in_channels=3)
        self.cls_head = GCNHead(num_classes=1, in_channels=self.backbone.out_channels)

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        batch, length, _ = windows.shape
        joints = windows[:, :, : KEYPOINTS * 3].reshape(batch, 1, length, KEYPOINTS, 3)
        return self.cls_head(self.backbone(joints)).squeeze(-1)


class FLASH(nn.Module):
    """FLASH (HyperMamba, Koffi 2026) rebuilt for the window's 17 keypoints (x, y,
    confidence; ``_vendor/flash``). Rows marked unobserved are zeroed; the score is the
    last frame's."""

    def __init__(self) -> None:
        super().__init__()
        self.net = HyperMamba()

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        batch, length, _ = windows.shape
        observed = windows[:, :, -1:]
        joints = (windows * observed)[:, :, : KEYPOINTS * 3].reshape(batch, length, KEYPOINTS, 3)
        return self.net(joints)[:, -1]


HEATMAP_SIZE = 32  # pixels per side (PoseC3D uses 56; reduced for CPU)
HEATMAP_EXTENT = 1.25  # the grid spans +-1.25 person heights around the reference
HEATMAP_SIGMA_PX = 0.6  # Gaussian width, as in PoseC3D


def keypoint_heatmaps(windows: torch.Tensor) -> torch.Tensor:
    """PoseC3D's input from pose windows: one Gaussian per keypoint and row on a
    ``HEATMAP_SIZE`` grid, scaled by the keypoint confidence; rows marked unobserved are
    empty. Output (batch, 17, length, size, size)."""
    batch, length, _ = windows.shape
    joints = windows[:, :, : KEYPOINTS * 3].reshape(batch, length, KEYPOINTS, 3)
    scale = (HEATMAP_SIZE - 1) / (2 * HEATMAP_EXTENT)
    centre = (HEATMAP_SIZE - 1) / 2
    x = joints[..., 0] * scale + centre  # (batch, length, 17) in pixels
    y = joints[..., 1] * scale + centre
    amplitude = joints[..., 2] * windows[:, :, -1:]
    grid = torch.arange(HEATMAP_SIZE, dtype=windows.dtype, device=windows.device)
    dx = (grid.view(1, 1, 1, 1, -1) - x.unsqueeze(-1).unsqueeze(-1)) ** 2
    dy = (grid.view(1, 1, 1, -1, 1) - y.unsqueeze(-1).unsqueeze(-1)) ** 2
    maps = torch.exp(-(dx + dy) / (2 * HEATMAP_SIGMA_PX**2))
    maps = maps * amplitude.unsqueeze(-1).unsqueeze(-1)
    return maps.permute(0, 2, 1, 3, 4)


class PoseC3D(nn.Module):
    """PoseC3D, reduced: MMAction2's SlowOnly ResNet3D-50 keypoint backbone (ported in
    ``_vendor/mmaction2``) over 32 x 32 heatmaps of the window's rows instead of 56 x 56
    heatmaps of 48 frames; ``backbone`` and ``cls_head`` keep MMAction2's names so its
    published checkpoints load."""

    def __init__(self) -> None:
        super().__init__()
        self.backbone = ResNet3dSlowOnly()
        self.cls_head = I3DHead(num_classes=1, in_channels=self.backbone.feat_dim)

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        return self.cls_head(self.backbone(keypoint_heatmaps(windows))).squeeze(-1)


def build_model(name: str, length: int, features: int, seed: int = 0) -> nn.Module:
    """A fresh model by name, initialised from ``seed``; ``length`` is for architectures
    that need it."""
    torch.manual_seed(seed)
    if name == "tcn":
        return SmallTCN(features)
    if name == "stgcnpp":
        return STGCNPP()
    if name == "tcnte":
        return TCNTE(features, length)
    if name == "flash":
        return FLASH()
    if name == "posec3d":
        return PoseC3D()
    raise ValueError(f"unknown model {name!r}")
