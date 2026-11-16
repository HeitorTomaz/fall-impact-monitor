"""The camera's mounting height (#78) reaches every fall rule: the service's, fim
analyze's (live and full-rate re-analysis), the replay's and fim eval's per dataset."""

import json
from pathlib import Path
from typing import Any

import pytest

from fall_impact_monitor.cli import analyze, evaluate
from fall_impact_monitor.cli.main import main
from fall_impact_monitor.config import Settings
from tests.cli.test_eval_cli import gmdcsa24
from tests.cli.test_replay import scripted  # noqa: F401  (fixture)


def spy(monkeypatch: pytest.MonkeyPatch) -> list[float | None]:
    seen: list[float | None] = []
    original = analyze.CameraAnalyzer

    def camera_analyzer(*args: Any, **kwargs: Any) -> Any:
        seen.append(kwargs.get("camera_height_m"))
        return original(*args, **kwargs)

    monkeypatch.setattr(analyze, "CameraAnalyzer", camera_analyzer)
    return seen


def test_analyze_and_replay_use_the_given_height(
    synthetic_video: Path,
    tmp_path: Path,
    scripted: None,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen = spy(monkeypatch)
    args = ["analyze", "--source", str(synthetic_video), "--out-dir", str(tmp_path)]
    assert main([*args, "--camera-height-m", "0.8"]) == 0
    assert main([*args]) == 0
    analyze.replay(tmp_path, Settings(), camera_height_m=2.1)
    assert seen == [0.8, 2.5, 2.1]  # unknown: the service's default


def test_analyze_rejects_an_implausible_height(synthetic_video: Path, tmp_path: Path) -> None:
    args = ["analyze", "--source", str(synthetic_video), "--out-dir", str(tmp_path)]
    with pytest.raises(SystemExit):
        main([*args, "--camera-height-m", "12"])


def test_eval_analyses_each_dataset_at_its_camera_height(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gmdcsa24(tmp_path / "data")
    heights: list[float] = []

    def fake_analysis(
        video: Path,
        out_dir: Path,
        manifest: Path,
        time_scale: float = 1.0,
        pad_end_s: float = 0.0,
        camera_height_m: float = 2.5,
    ) -> Path:
        heights.append(camera_height_m)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "events.jsonl").write_text("")
        return out_dir / "events.jsonl"

    monkeypatch.setattr(evaluate, "run_analysis", fake_analysis)
    args = ["eval", "--dataset", "gmdcsa24", "--root", str(tmp_path / "data")]
    assert main([*args, "--out-dir", str(tmp_path / "a")]) == 0
    assert main([*args, "--out-dir", str(tmp_path / "b"), "--camera-height-m", "2.4"]) == 0
    assert heights == [evaluate.CAMERA_HEIGHT_M["gmdcsa24"]] * 2 + [2.4] * 2
    report = json.loads((tmp_path / "a" / "report.json").read_text())
    assert report["dataset"] == "gmdcsa24"
