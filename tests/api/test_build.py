import json
from pathlib import Path

import numpy as np
import pytest

from fall_impact_monitor.api import build_pipeline
from fall_impact_monitor.api.perception import PerceptionPipeline
from fall_impact_monitor.cli.main import main
from fall_impact_monitor.config import Settings
from fall_impact_monitor.perception.cadence import CadencePolicy, RateMonitor
from fall_impact_monitor.perception.detector import Detection
from fall_impact_monitor.perception.models import ModelError
from fall_impact_monitor.perception.pose import Pose
from fall_impact_monitor.perception.tracking import PersonTracker

REPO = Path(__file__).parents[2]


def test_build_pipeline_names_a_missing_model(tmp_path: Path) -> None:
    settings = Settings(_env_file=None, models_dir=str(tmp_path))
    with pytest.raises(ModelError, match="yolox_m_humanart"):
        build_pipeline(settings, manifest_path=REPO / "models.toml", started_s=0.0)


class OnePerson:
    def detect(self, image_bgr: np.ndarray) -> list[Detection]:
        return [Detection((100.0, 50.0, 200.0, 300.0), 0.9)]


class FlatPose:
    def estimate(self, image_bgr: np.ndarray, boxes: object) -> list[Pose | None]:
        return [Pose(np.full((17, 2), 150.0, np.float32), np.ones(17, np.float32))]


def test_analyze_writes_tracks_and_overlay(
    synthetic_video: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fall_impact_monitor.cli import analyze

    def fake_build(settings: Settings, **kwargs: object) -> PerceptionPipeline:
        return PerceptionPipeline(
            OnePerson(),
            FlatPose(),
            PersonTracker(min_age_frames=1),
            CadencePolicy(),
            RateMonitor(min_fps=0.5, started_s=0.0),
        )

    monkeypatch.setattr(analyze, "build_pipeline", fake_build)
    code = main(
        ["analyze", "--source", str(synthetic_video), "--out-dir", str(tmp_path), "--overlay"]
    )
    assert code == 0
    lines = [json.loads(line) for line in (tmp_path / "tracks.jsonl").read_text().splitlines()]
    assert lines and all({"seq", "arrival_s", "tracks"} <= set(line) for line in lines)
    assert any(line["tracks"] for line in lines)
    assert (tmp_path / "overlay.mp4").stat().st_size > 0


def test_analyze_labels_frames_with_their_own_presentation_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fall_impact_monitor.cli import analyze

    from ..conftest import write_synthetic_video

    video = write_synthetic_video(tmp_path / "bframes.mp4", b_frames=2)

    def fake_build(settings: Settings, **kwargs: object) -> PerceptionPipeline:
        return PerceptionPipeline(
            OnePerson(),
            FlatPose(),
            PersonTracker(min_age_frames=1),
            CadencePolicy(empty_fps=25.0, person_fps=25.0),
            RateMonitor(min_fps=0.5, started_s=0.0),
        )

    monkeypatch.setattr(analyze, "build_pipeline", fake_build)
    assert main(["analyze", "--source", str(video), "--out-dir", str(tmp_path / "out")]) == 0
    lines = [
        json.loads(line) for line in (tmp_path / "out" / "tracks.jsonl").read_text().splitlines()
    ]
    times = [line["pts_s"] for line in lines]
    assert len(lines) == 75
    assert times == sorted(times) and len(set(times)) == len(times)
