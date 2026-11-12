import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from fall_impact_monitor.api import build_pipeline
from fall_impact_monitor.api.perception import FrameResult, PerceptionPipeline
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
    assert lines and all(
        {"seq", "arrival_s", "tracks", "detections"} <= set(line) for line in lines
    )
    assert lines[0]["detections"][0]["score"] == 0.9
    assert any(line["tracks"] for line in lines)
    assert (tmp_path / "overlay.mp4").stat().st_size > 0
    events = (tmp_path / "events.jsonl").read_text().splitlines()
    assert all(json.loads(line)["category"] == "possible_fall" for line in events)


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


def test_analyze_rotates_frames_like_the_service(
    synthetic_video: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fall_impact_monitor.cli import analyze

    seen: list[NDArray[np.uint8]] = []

    class Recording:
        def process(self, image: NDArray[np.uint8], record: object) -> FrameResult:
            seen.append(image.copy())
            return FrameResult(record, [], analysed=False)  # type: ignore[arg-type]

    monkeypatch.setattr(analyze, "build_pipeline", lambda settings, **kw: Recording())
    args = ["analyze", "--source", str(synthetic_video), "--max-frames", "1"]
    assert main([*args, "--out-dir", str(tmp_path / "a")]) == 0
    assert main([*args, "--out-dir", str(tmp_path / "b"), "--rotate-deg", "180"]) == 0
    assert np.array_equal(seen[1], seen[0][::-1, ::-1])


def test_analyze_confirms_a_near_miss_at_full_rate(
    synthetic_video: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The live pass sees a stumble (a near-miss); every frame shows a fall."""
    from fall_impact_monitor.cli import analyze
    from fall_impact_monitor.perception.tracking import Track
    from tests.analysis.test_fall_analyzer import HIP_UP, falling, make_track, standing

    def stumble(seq: int) -> list[Track]:
        if 25 <= seq < 35:
            return [make_track(0, HIP_UP + 30.0, 60.0, 0.6)]
        return [standing()]

    def fall(seq: int) -> list[Track]:
        if seq < 25:
            return [standing()]
        return [falling(0, min(1.0, (seq - 24) / 10))]

    class Scripted:
        def __init__(self, script: Any, every: int) -> None:
            self.script, self.every = script, every

        def process(self, image: Any, rec: Any) -> FrameResult:
            if rec.seq % self.every:
                return FrameResult(rec, [], analysed=False)
            return FrameResult(rec, self.script(rec.seq), analysed=True)

        def sharing_models(self, tracker: Any) -> "Scripted":
            return Scripted(fall, every=1)

    monkeypatch.setattr(analyze, "build_pipeline", lambda settings, **kw: Scripted(stumble, 3))
    assert main(["analyze", "--source", str(synthetic_video), "--out-dir", str(tmp_path)]) == 0
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    falls = [e for e in events if e["category"] == "possible_fall"]
    assert any(e["near_miss"] for e in falls)  # the live candidate is kept
    confirmed = [e for e in falls if not e["near_miss"]]
    assert len(confirmed) == 1 and confirmed[0]["evidence"]["reanalysed_at_full_rate"] is True
