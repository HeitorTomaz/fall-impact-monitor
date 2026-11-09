import subprocess
import sys
from datetime import UTC, datetime
from fractions import Fraction

import numpy as np

from fall_impact_monitor.api import FrameResult, PerceptionPipeline
from fall_impact_monitor.capture import FrameRecord, TimestampQuality
from fall_impact_monitor.perception.cadence import CadencePolicy, RateMonitor
from fall_impact_monitor.perception.detector import Detection
from fall_impact_monitor.perception.pose import Pose
from fall_impact_monitor.perception.tracking import PersonTracker

FRAME = np.zeros((120, 160, 3), np.uint8)


def record(t_s: float, seq: int = 0) -> FrameRecord:
    return FrameRecord(
        "cam1",
        seq,
        seq * 3000,
        Fraction(1, 90000),
        int(t_s * 1e9),
        datetime(2026, 1, 1, tzinfo=UTC),
        False,
        100,
        TimestampQuality.OK,
    )


class FakeDetector:
    def __init__(self, people: bool) -> None:
        self.people = people
        self.calls = 0

    def detect(self, image_bgr: np.ndarray) -> list[Detection]:
        self.calls += 1
        return [Detection((10.0, 10.0, 50.0, 110.0), 0.9)] if self.people else []


class FakePose:
    def __init__(self) -> None:
        self.calls = 0

    def estimate(self, image_bgr: np.ndarray, boxes: object) -> list[Pose | None]:
        self.calls += 1
        return [Pose(np.zeros((17, 2), np.float32), np.ones(17, np.float32))]


def pipeline(people: bool) -> tuple[PerceptionPipeline, FakeDetector, FakePose]:
    detector, pose = FakeDetector(people), FakePose()
    return (
        PerceptionPipeline(
            detector,
            pose,
            PersonTracker(),
            CadencePolicy(empty_fps=1.0, person_fps=5.0),
            RateMonitor(min_fps=0.5, started_s=0.0),
        ),
        detector,
        pose,
    )


def feed(pipe: PerceptionPipeline, seconds: float, fps: float = 25.0) -> list[FrameResult]:
    return [pipe.process(FRAME, record(i / fps, i)) for i in range(int(seconds * fps))]


def test_empty_scene_is_analysed_at_the_empty_rate_without_pose() -> None:
    pipe, detector, pose = pipeline(people=False)
    results = feed(pipe, 4.0)
    assert sum(r.analysed for r in results) == 4  # 1 fps
    assert detector.calls == 4 and pose.calls == 0
    assert all(r.tracks == [] for r in results)


def test_person_switches_to_the_person_rate_and_runs_pose() -> None:
    pipe, _, pose = pipeline(people=True)
    results = feed(pipe, 4.0)
    analysed = [r for r in results if r.analysed]
    assert 17 <= len(analysed) <= 21  # ~5 fps after the first analysed frame
    assert pose.calls == len(analysed)
    assert analysed[-1].tracks and analysed[-1].tracks[0].keypoints is not None


def test_attention_analyses_every_frame_until_it_expires() -> None:
    pipe, _, _ = pipeline(people=True)
    pipe.request_attention(until_s=1.0)
    results = feed(pipe, 2.0)
    assert all(r.analysed for r in results[:25])
    assert sum(r.analysed for r in results[25:]) <= 6


def test_skipped_frames_report_not_analysed() -> None:
    pipe, _, _ = pipeline(people=False)
    results = feed(pipe, 1.0)
    assert results[0].analysed and not results[1].analysed
    assert results[1].record.seq == 1


def test_degraded_when_analysis_stops() -> None:
    pipe, _, _ = pipeline(people=False)
    feed(pipe, 2.0)
    assert not pipe.degraded(5.0)
    assert pipe.degraded(30.0)


def test_detection_core_does_not_import_app_layers() -> None:
    code = (
        "import sys, fall_impact_monitor.api\n"
        "bad = [m for m in sys.modules if m.split('.')[0] in ('aiogram', 'sqlalchemy') or "
        "m.startswith(('fall_impact_monitor.sinks', 'fall_impact_monitor.storage', "
        "'fall_impact_monitor.recording', 'fall_impact_monitor.health'))]\n"
        "assert not bad, bad\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
