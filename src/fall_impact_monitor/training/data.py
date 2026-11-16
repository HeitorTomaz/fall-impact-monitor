"""Window sets, splits and window-level metrics for training (#15). numpy only."""

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from fall_impact_monitor.evaluation.datasets import DatasetEntry, split_of


@dataclass(frozen=True)
class WindowSet:
    features: NDArray[np.float32]  # (n, length, features)
    labels: NDArray[np.int8]
    groups: NDArray[np.str_]
    videos: NDArray[np.str_]
    datasets: NDArray[np.str_]
    perception: Any  # the perception configuration of the analyses

    def subset(self, rows: NDArray[np.bool_]) -> "WindowSet":
        return WindowSet(
            self.features[rows],
            self.labels[rows],
            self.groups[rows],
            self.videos[rows],
            self.datasets[rows],
            self.perception,
        )


def load_windows(paths: Sequence[Path]) -> WindowSet:
    """Join ``fim windows`` files made with the same perception configuration."""
    columns: list[list[NDArray[Any]]] = [[], [], [], [], []]
    perception: Any = None
    for index, path in enumerate(paths):
        with np.load(path, allow_pickle=False) as saved:
            meta = json.loads(str(saved["meta"]))
            if index == 0:
                perception = meta["perception"]
            elif meta["perception"] != perception:
                raise ValueError(f"{path.name}: analysed with another perception configuration")
            rows = len(saved["labels"])
            for column, values in zip(
                columns,
                (
                    saved["features"],
                    saved["labels"],
                    saved["groups"],
                    saved["videos"],
                    np.full(rows, meta["dataset"]),
                ),
                strict=True,
            ):
                column.append(values)
    features, labels, groups, videos, datasets = (np.concatenate(c) for c in columns)
    return WindowSet(features, labels, groups, videos, datasets, perception)


def split_rows(
    windows: WindowSet, manifest: Mapping[str, DatasetEntry]
) -> dict[str, NDArray[np.bool_]]:
    """Rows of each split (train, validation, test) by the manifest's splits, which keep
    a person or an event on one side."""
    names = [
        split_of(manifest[str(d)], str(v))
        for d, v in zip(windows.datasets, windows.videos, strict=True)
    ]
    missing = sorted({str(v) for v, s in zip(windows.videos, names, strict=True) if s is None})
    if missing:
        raise ValueError("videos outside every split: " + ", ".join(missing[:5]))
    array = np.array(names, dtype=object)
    return {split: array == split for split in ("train", "validation", "test")}


def average_precision(labels: NDArray[np.integer], scores: NDArray[np.floating]) -> float:
    """Mean precision at the rank of each positive, ranked by decreasing score."""
    order = np.argsort(-np.asarray(scores, np.float64), kind="stable")
    ranked = np.asarray(labels)[order] == 1
    if not ranked.any():
        return 0.0
    precision = np.cumsum(ranked) / np.arange(1, len(ranked) + 1)
    return float(precision[ranked].mean())
