"""`fim train` (#15). The end-to-end test needs the train group."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from fall_impact_monitor.cli.main import main
from fall_impact_monitor.evaluation.windows import Window, save_windows
from tests.training.windows_helpers import synthetic


def windows_file(path: Path, rows_per_subject: int = 60) -> Path:
    windows, groups, videos = [], [], []
    for subject in (1, 2, 3, 4):  # train 1-2, validation 3, test 4 (datasets.toml)
        x, y = synthetic(rows_per_subject, seed=subject)
        windows += [Window(f, int(label), 1, 0.0) for f, label in zip(x, y, strict=True)]
        groups += [f"gmdcsa24:Subject {subject}"] * rows_per_subject
        videos += [f"Subject {subject}/Fall/01.mp4"] * rows_per_subject
    save_windows(path, windows, groups, videos, {"dataset": "gmdcsa24", "perception": {}})
    return path


def test_train_writes_an_onnx_model_and_its_test_scores(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    data = windows_file(tmp_path / "w.npz")
    out = tmp_path / "tcn"
    args = ["train", "--windows", str(data), "--model", "tcn", "--epochs", "5"]
    assert main([*args, "--out-dir", str(out)]) == 0
    assert (out / "model.onnx").stat().st_size > 0
    metrics = json.loads((out / "metrics.json").read_text())
    assert metrics["model"] == "tcn" and metrics["seed"] == 0
    assert metrics["windows"] == {"train": 120, "validation": 60, "test": 60}
    assert 0.0 <= metrics["test_ap"] <= 1.0
    assert set(metrics["test_ap_by_dataset"]) == {"gmdcsa24"}
    assert len(metrics["inputs"][0]["sha256"]) == 64


def test_train_says_how_to_get_pytorch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setitem(sys.modules, "torch", None)  # as where the train group is absent
    for name in ("fall_impact_monitor.training.models", "fall_impact_monitor.training.train"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    data = windows_file(tmp_path / "w.npz", rows_per_subject=4)
    assert main(["train", "--windows", str(data), "--out-dir", str(tmp_path / "o")]) == 2
    assert "--group train" in capsys.readouterr().out
    assert np.load(data)["labels"].size == 16


def test_train_fine_tunes_from_a_verified_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("torch")
    from fall_impact_monitor.training import pretrained
    from tests.training.test_stgcn import checkpoint

    models = tmp_path / "models"
    models.mkdir()
    digest = checkpoint(models / "fake_ntu.pth")
    fake = pretrained.Pretrained("stgcnpp", "https://example.org/fake_ntu.pth", digest)
    monkeypatch.setitem(pretrained.PRETRAINED, "fake_ntu", fake)
    data = windows_file(tmp_path / "w.npz", rows_per_subject=8)
    args = ["train", "--windows", str(data), "--model", "stgcnpp", "--epochs", "1"]
    args += ["--pretrained", "fake_ntu", "--models-dir", str(models)]
    assert main([*args, "--out-dir", str(tmp_path / "o")]) == 0
    metrics = json.loads((tmp_path / "o" / "metrics.json").read_text())
    assert metrics["pretrained"] == {"name": "fake_ntu", "sha256": digest}


def test_train_names_the_checkpoint_to_download(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pytest.importorskip("torch")
    data = windows_file(tmp_path / "w.npz", rows_per_subject=4)
    args = ["train", "--windows", str(data), "--model", "stgcnpp"]
    args += ["--pretrained", "stgcnpp_ntu60_2d_joint", "--models-dir", str(tmp_path)]
    assert main([*args, "--out-dir", str(tmp_path / "o")]) == 2
    out = capsys.readouterr().out
    assert "download.openmmlab.com" in out and "86e1e77a" in out
