"""Evaluation crop (#93): UR Fall videos hold the depth map and the RGB image side by
side; only the RGB half is analysed, live and in the full-rate re-analysis."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from fall_impact_monitor.api.confirmation import _bgr
from fall_impact_monitor.cli import analyze, evaluate
from fall_impact_monitor.cli.main import main
from tests.conftest import write_synthetic_video


def test_a_frame_is_cropped_before_anything_else() -> None:
    frame = np.zeros((240, 640, 3), np.uint8)
    frame[:, 320:] = 255  # the right half
    image = _bgr(frame, 0, 10_000, crop_px=(320, 0, 320, 240))
    assert image.shape == (240, 320, 3) and int(image.min()) == 255


def test_analyze_gives_the_pipeline_the_cropped_frames(
    synthetic_video: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.cli.test_replay import Scripted, stumble

    monkeypatch.setattr(analyze, "build_pipeline", lambda settings, **kw: Scripted(stumble, 3))
    seen: list[tuple[int, ...]] = []
    original = analyze._Analyzer._process

    def process(self: Any, image: Any, record: Any, *, padded: bool) -> None:
        seen.append(image.shape)
        original(self, image, record, padded=padded)

    monkeypatch.setattr(analyze._Analyzer, "_process", process)
    args = ["analyze", "--source", str(synthetic_video), "--out-dir", str(tmp_path)]
    assert main([*args, "--crop-px", "320,0,320,360"]) == 0
    assert seen and all(s[:2] == (360, 320) for s in seen)
    lines = [json.loads(x) for x in (tmp_path / "tracks.jsonl").read_text().splitlines()]
    assert all(line["size"] == [320, 360] for line in lines if line["analysed"])


def test_eval_crops_the_rgb_half_of_ur_fall_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "urfd"
    (root / "raw" / "labels").mkdir(parents=True)
    (root / "raw" / "videos").mkdir(parents=True)
    for name in ("fall-01-cam0.mp4", "adl-01-cam0.mp4"):
        write_synthetic_video(root / "raw" / "videos" / name)  # 640 x 360
    rows = "".join(f"fall-01,{i + 1},{-1 if i < 3 else 0},0,0,0,0,0,0,0,0\n" for i in range(10))
    (root / "raw" / "labels" / "urfall-cam0-falls.csv").write_text(rows)
    (root / "raw" / "labels" / "urfall-cam0-adls.csv").write_text(rows.replace("fall-01", "adl-01"))
    crops: list[Any] = []

    def fake_analysis(
        video: Path, out_dir: Path, manifest: Path, *args: Any, **kwargs: Any
    ) -> Path:
        crops.append(kwargs.get("crop_px"))
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "events.jsonl").write_text("")
        return out_dir / "events.jsonl"

    monkeypatch.setattr(evaluate, "run_analysis", fake_analysis)
    out = tmp_path / "report"
    assert main(["eval", "--dataset", "urfd", "--root", str(root), "--out-dir", str(out)]) == 0
    assert crops == [(320, 0, 320, 360)] * 2
