"""Temporal classifier architectures (#15). Input ``(batch, length, features)`` pose
windows (``analysis.pose_window``); output one fall logit per window."""

import torch
from torch import nn

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
    raise ValueError(f"unknown model {name!r}")
