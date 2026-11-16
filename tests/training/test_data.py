"""Window sets and splits for training (#15): numpy only, no PyTorch needed."""

import json
from pathlib import Path

import numpy as np
import pytest

from fall_impact_monitor.evaluation.datasets import load_dataset_manifest
from fall_impact_monitor.evaluation.windows import Window, save_windows
from fall_impact_monitor.training.data import average_precision, load_windows, split_rows

ROOT = Path(__file__).resolve().parents[2]


def write(path: Path, dataset: str, videos: list[str], perception: dict[str, float]) -> Path:
    windows = [
        Window(np.zeros((16, 53), np.float32), i % 2, 1, float(i)) for i in range(len(videos))
    ]
    groups = [f"{dataset}:{v.split('/')[0]}" for v in videos]
    save_windows(path, windows, groups, videos, {"dataset": dataset, "perception": perception})
    return path


def test_window_files_are_joined_and_remember_their_dataset(tmp_path: Path) -> None:
    a = write(tmp_path / "a.npz", "gmdcsa24", ["Subject 1/Fall/01.mp4"] * 2, {"x": 1.0})
    b = write(tmp_path / "b.npz", "mcfd", ["dataset/chute01/cam1.avi"] * 3, {"x": 1.0})
    windows = load_windows([a, b])
    assert windows.features.shape == (5, 16, 53)
    assert windows.datasets.tolist() == ["gmdcsa24"] * 2 + ["mcfd"] * 3


def test_windows_from_different_perception_are_not_mixed(tmp_path: Path) -> None:
    a = write(tmp_path / "a.npz", "gmdcsa24", ["Subject 1/Fall/01.mp4"], {"x": 1.0})
    b = write(tmp_path / "b.npz", "mcfd", ["dataset/chute01/cam1.avi"], {"x": 2.0})
    with pytest.raises(ValueError, match="perception"):
        load_windows([a, b])


def test_rows_follow_the_manifest_splits(tmp_path: Path) -> None:
    videos = ["Subject 1/Fall/01.mp4", "Subject 3/ADL/02.mp4", "Subject 4/Fall/05.mp4"]
    windows = load_windows([write(tmp_path / "a.npz", "gmdcsa24", videos, {})])
    splits = split_rows(windows, load_dataset_manifest(ROOT / "datasets.toml"))
    assert splits["train"].tolist() == [True, False, False]
    assert splits["validation"].tolist() == [False, True, False]
    assert splits["test"].tolist() == [False, False, True]


def test_a_video_outside_every_split_is_an_error(tmp_path: Path) -> None:
    windows = load_windows([write(tmp_path / "a.npz", "gmdcsa24", ["Subject 9/x.mp4"], {})])
    with pytest.raises(ValueError, match="Subject 9"):
        split_rows(windows, load_dataset_manifest(ROOT / "datasets.toml"))


def test_average_precision() -> None:
    labels = np.array([0, 1, 0, 1])
    assert average_precision(labels, np.array([0.1, 0.9, 0.2, 0.8])) == pytest.approx(1.0)
    assert average_precision(labels, np.array([0.9, 0.1, 0.8, 0.2])) == pytest.approx(
        (1 / 3 + 2 / 4) / 2
    )
    assert json.dumps(average_precision(labels, np.zeros(4)))  # a plain float
