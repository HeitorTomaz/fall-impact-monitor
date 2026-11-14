"""Temporal classifier architectures (#15). Input ``(batch, length, features)`` pose
windows (``analysis.pose_window``); output one fall logit per window."""

import torch
from torch import nn


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


def build_model(name: str, length: int, features: int, seed: int = 0) -> nn.Module:
    """A fresh model by name, initialised from ``seed``; ``length`` is for architectures
    that need it."""
    torch.manual_seed(seed)
    if name == "tcn":
        return SmallTCN(features)
    raise ValueError(f"unknown model {name!r}")
