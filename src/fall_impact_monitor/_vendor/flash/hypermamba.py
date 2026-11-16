# Copyright (c) 2026 Tresor Y. Koffi
# Copyright 2026 The Fall Impact Monitor contributors
# SPDX-License-Identifier: MIT
#
# Adapted from https://github.com/Tresor-Koffi/FLASH-Impact-Fall-Detection
#   @04a1215314472085aea68009159c5f9237e9e7cf
#   HGCN/hypergraph.py (Hypergraph: hyperedges, normalised incidence)
#   HGCN/hmamba.py (HypergraphConv, SelectiveSSM, MambaBlock, HyperMamba.forward)
# changed: hyperedges rebuilt for the 17 COCO keypoints (2D x, y, confidence) instead
#   of 33 3D MediaPipe joints, keeping the same biomechanical groups; smaller sizes
#   (32 features per keypoint, 64 states instead of 128 and 256: the selective scan's
#   state tensor would otherwise exceed 2 GB per batch on CPU); the window's last frame
#   is read out instead of one logit per frame; rows marked unobserved are zeroed;
#   training, metrics and file handling removed; type hints.
"""FLASH / HyperMamba (Koffi, 2026): hypergraph convolution over the keypoints, a
selective state-space (Mamba-style) block over time, multi-scale temporal convolutions.
The selective scan is the upstream pure-PyTorch loop (no CUDA kernels)."""

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

# COCO-17: 0 nose, 1-2 eyes, 3-4 ears, 5-6 shoulders, 7-8 elbows, 9-10 wrists,
# 11-12 hips, 13-14 knees, 15-16 ankles (left first).
COCO_HYPEREDGES = [
    [0, 1, 2, 3, 4],  # head
    [3, 4],  # ears
    [5, 6, 11, 12],  # shoulder-hip chain (force transmission)
    [5, 6],  # shoulders
    [11, 12],  # hips
    [5, 7, 9],  # left arm (protective reflex)
    [7, 9],  # left elbow-wrist
    [6, 8, 10],  # right arm
    [8, 10],  # right elbow-wrist
    [11, 13, 15],  # left leg
    [13, 15],  # left knee-ankle
    [12, 14, 16],  # right leg
    [14, 16],  # right knee-ankle
    [5, 12],  # left shoulder - right hip diagonal
    [6, 11],  # right shoulder - left hip diagonal
    [0, 9, 10, 13, 14],  # forward-fall impact points: nose, wrists, knees
    [11, 12, 15, 16],  # backward-fall impact points: hips, ankles
    [5, 11, 13, 15],  # left side
    [6, 12, 14, 16],  # right side
    [0, 5, 6, 11, 12, 13, 14],  # head to knees
    [15, 16],  # feet
]


def coco_hypergraph() -> tuple[torch.Tensor, torch.Tensor]:
    """Incidence matrix H (17 x edges) and H_norm = Dv^-1/2 H De^-1 H^T Dv^-1/2."""
    num_node = 17
    H = np.zeros((num_node, len(COCO_HYPEREDGES)), dtype=np.float32)
    for e_idx, edge in enumerate(COCO_HYPEREDGES):
        for v in edge:
            H[v, e_idx] = 1.0
    eps = 1e-8
    d_v = np.sum(H, axis=1)
    d_e = np.sum(H, axis=0)
    D_v_inv_sqrt = np.diag(1.0 / np.sqrt(d_v + eps))
    D_e_inv = np.diag(1.0 / (d_e + eps))
    H_norm = D_v_inv_sqrt @ H @ D_e_inv @ H.T @ D_v_inv_sqrt
    return torch.tensor(H), torch.tensor(H_norm, dtype=torch.float32)


class HypergraphConv(nn.Module):
    """X' = relu(H_norm X W)."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        nn.init.kaiming_normal_(self.linear.weight, nonlinearity="relu")

    def forward(self, x: torch.Tensor, H_norm: torch.Tensor) -> torch.Tensor:
        x = torch.einsum("ij,btjf->btif", H_norm, x)
        return F.relu(self.linear(x))


class SelectiveSSM(nn.Module):
    """Selective state-space model: h_k = exp(delta A) h_{k-1} + B x_k, y_k = C h_k + D x_k."""

    def __init__(self, d_model: int, d_state: int, dropout_rate: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.in_proj = nn.Linear(d_model, d_model * 2)
        self.A_proj = nn.Linear(d_model, d_state)
        self.B_proj = nn.Linear(d_model, d_state)
        self.C_proj = nn.Linear(d_model, d_state)
        self.D = nn.Parameter(torch.ones(d_model))
        self.delta_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout_rate)
        nn.init.kaiming_normal_(self.in_proj.weight, nonlinearity="relu")
        nn.init.kaiming_normal_(self.out_proj.weight, nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        x_main, x_gate = self.in_proj(x).chunk(2, dim=-1)
        A = -F.softplus(self.A_proj(x_main))
        B = self.B_proj(x_main)
        C = self.C_proj(x_main)
        delta = F.softplus(self.delta_proj(x_main))
        deltaA = delta.unsqueeze(-1) * A.unsqueeze(-2)
        h = torch.zeros(batch_size, self.d_model, self.d_state, device=x.device)
        outputs = []
        for t in range(seq_len):
            h = h * torch.exp(deltaA[:, t]) + B[:, t].unsqueeze(1) * x_main[:, t].unsqueeze(-1)
            outputs.append((h * C[:, t].unsqueeze(1)).sum(dim=-1))
        y = torch.stack(outputs, dim=1) + x_main * self.D
        y = self.out_proj(y * F.silu(x_gate))
        return self.dropout(y)


class MambaBlock(nn.Module):
    def __init__(self, d_model: int, d_state: int, dropout_rate: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ssm = SelectiveSSM(d_model, d_state, dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.ssm(self.norm(x))


class HyperMamba(nn.Module):
    """Input (B, T, 17, 3); output one logit per frame (B, T)."""

    def __init__(
        self, hgcn_hidden: int = 32, mamba_d_state: int = 64, dropout_rate: float = 0.3
    ) -> None:
        super().__init__()
        num_nodes = 17
        H, H_norm = coco_hypergraph()
        self.register_buffer("H", H)
        self.register_buffer("H_norm", H_norm)
        self.hgcn1 = HypergraphConv(3, hgcn_hidden)
        self.hgcn2 = HypergraphConv(hgcn_hidden, hgcn_hidden)
        self.mamba = MambaBlock(num_nodes * hgcn_hidden, mamba_d_state, dropout_rate)
        self.conv_z1 = nn.Conv2d(hgcn_hidden, 64, kernel_size=(9, 1), padding=(4, 0))
        self.conv_z2 = nn.Conv2d(64, 64, kernel_size=(15, 1), padding=(7, 0))
        self.pad_z3 = nn.ZeroPad2d((0, 0, 10, 9))
        self.conv_z3 = nn.Conv2d(64, 64, kernel_size=(20, 1))
        self.dropout = nn.Dropout(dropout_rate)
        self.conv_final_1 = nn.Conv2d(192, 64, kernel_size=(1, 1))
        self.conv_final_2 = nn.Conv2d(64, 32, kernel_size=(1, num_nodes))
        self.dense1 = nn.Linear(32, 32)
        self.dense2 = nn.Linear(32, 1)
        self.dropout_final = nn.Dropout(0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, N, _ = x.shape
        x = self.hgcn1(x, self.H_norm)
        x = self.hgcn2(x, self.H_norm)
        x = self.mamba(x.reshape(B, T, -1)).reshape(B, T, N, -1)
        x = x.permute(0, 3, 1, 2).contiguous()
        z1 = self.dropout(F.relu(self.conv_z1(x)))
        z2 = self.dropout(F.relu(self.conv_z2(z1)))
        z3 = self.dropout(F.relu(self.conv_z3(self.pad_z3(z2))))
        z = torch.cat([z1, z2, z3], dim=1)
        x = self.conv_final_2(F.relu(self.conv_final_1(z)))
        x = F.relu(self.dense1(x.squeeze(-1).permute(0, 2, 1)))
        return self.dense2(self.dropout_final(x)).squeeze(-1)
