"""Published pretrained weights for fine-tuning (#15).

Openly downloadable checkpoints may be fine-tuned privately (maintainer decision,
2026-11-13); their terms are in ``MODELS.md`` and fine-tuned weights are never
published. A file is used only when its SHA-256 matches, and it is read with PyTorch's
weights-only loader, which does not run code from the file.
"""

import hashlib
from dataclasses import dataclass
from pathlib import Path

import torch

from fall_impact_monitor.training.models import STGCNPP, PoseC3D


@dataclass(frozen=True)
class Pretrained:
    model: str  # architecture in training.models
    url: str
    sha256: str


PRETRAINED = {
    "stgcnpp_ntu60_2d_joint": Pretrained(
        "stgcnpp",
        "https://download.openmmlab.com/mmaction/v1.0/skeleton/stgcnpp/"
        "stgcnpp_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d/"
        "stgcnpp_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d_20221228-86e1e77a.pth",
        "86e1e77a7b27dd53c6ccc1600a4cc3c321ec8e04d17f25a41fbbb7896026945a",
    ),
    "posec3d_ntu60_2d_keypoint": Pretrained(
        "posec3d",
        "https://download.openmmlab.com/mmaction/v1.0/skeleton/posec3d/"
        "slowonly_r50_8xb16-u48-240e_ntu60-xsub-keypoint/"
        "slowonly_r50_8xb16-u48-240e_ntu60-xsub-keypoint_20220815-38db104b.pth",
        "38db104be0139bc2d57d662031de2e59eaa5302b6d33548628de1bf4af2a0f9d",
    ),
}


def load_pretrained(model: STGCNPP | PoseC3D, path: Path, sha256: str) -> None:
    """Load an MMAction2 checkpoint's ``backbone`` into ``model``; the classification
    head (60 action classes) is not loaded: the model keeps its own fall output."""
    backbone_module: torch.nn.Module = model.backbone
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != sha256:
        raise ValueError(f"{path.name} does not match its sha256")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    state = checkpoint.get("state_dict", checkpoint)
    backbone = {k[len("backbone.") :]: v for k, v in state.items() if k.startswith("backbone.")}
    expected = set(backbone_module.state_dict())
    if set(backbone) != expected:
        missing = sorted(expected - set(backbone))[:3]
        raise ValueError(f"{path.name}: backbone weights do not match the model ({missing})")
    backbone_module.load_state_dict(backbone)
