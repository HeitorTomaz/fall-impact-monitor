"""Fitting, scoring and ONNX export of temporal classifiers (#15)."""

import copy
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn

from fall_impact_monitor.training.data import average_precision

Arrays = tuple[NDArray[np.float32], NDArray[np.integer]]


@dataclass(frozen=True)
class FitResult:
    best_epoch: int
    validation_ap: float  # window-level average precision of the kept weights
    history: list[float] = field(default_factory=list)  # validation AP per epoch


def predict(
    model: nn.Module, features: NDArray[np.float32], batch: int = 512
) -> NDArray[np.float32]:
    """Fall probability of each window."""
    model.eval()
    scores = []
    with torch.no_grad():
        for start in range(0, len(features), batch):
            chunk = torch.from_numpy(np.ascontiguousarray(features[start : start + batch]))
            scores.append(torch.sigmoid(model(chunk)).numpy())
    return np.concatenate(scores).astype(np.float32) if scores else np.zeros(0, np.float32)


def fit(
    model: nn.Module,
    train: Arrays,
    validation: Arrays,
    *,
    epochs: int = 30,
    seed: int = 0,
    learning_rate: float = 1e-3,
    batch: int = 64,
) -> FitResult:
    """Train with falls weighted by their rarity; keep the epoch with the best
    validation average precision."""
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    x = torch.from_numpy(np.ascontiguousarray(train[0]))
    y = torch.from_numpy(np.asarray(train[1], np.float32))
    positives = float(y.sum())
    pos_weight = torch.tensor((len(y) - positives) / max(positives, 1.0))
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best_ap, best_epoch, best_state = -1.0, 0, copy.deepcopy(model.state_dict())
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(y), generator=generator)
        for start in range(0, len(y), batch):
            rows = order[start : start + batch]
            if len(rows) < 2:  # batch normalisation needs two windows
                continue
            optimizer.zero_grad()
            loss_fn(model(x[rows]), y[rows]).backward()
            optimizer.step()
        ap = average_precision(validation[1], predict(model, validation[0]))
        history.append(ap)
        if ap > best_ap:
            best_ap, best_epoch, best_state = ap, epoch, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return FitResult(best_epoch, best_ap, history)


class _Scores(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.model(windows))


def export_onnx(model: nn.Module, path: Path, length: int, features: int) -> Path:
    """ONNX model with input ``windows`` (batch, length, features) and output ``scores``
    (fall probabilities), any batch size."""
    model.eval()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        _Scores(model),
        (torch.zeros(2, length, features),),
        str(path),
        input_names=["windows"],
        output_names=["scores"],
        dynamic_shapes={"windows": {0: torch.export.Dim.DYNAMIC}},
    )
    return path
