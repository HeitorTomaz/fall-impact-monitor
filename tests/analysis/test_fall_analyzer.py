import math
from datetime import UTC, datetime
from fractions import Fraction

import numpy as np
import pytest

from fall_impact_monitor.analysis.fall import FallAnalyzer, fall_confidence
from fall_impact_monitor.api.perception import FrameResult
from fall_impact_monitor.capture import FrameRecord, TimestampQuality
from fall_impact_monitor.events import EventCategory
from fall_impact_monitor.perception.tracking import Track

FPS = 5.0
HEIGHT = 200.0
HIP_UP = 150.0


def make_track(
    track_id: int,
    hip: float,
    angle: float,
    aspect: float,
    height: float = HEIGHT,
    identity: str = "stable",
) -> Track:
    torso = 60.0
    dx, dy = torso * math.sin(math.radians(angle)), torso * math.cos(math.radians(angle))
    cx = 300.0 + 200.0 * track_id
    points = np.zeros((17, 2), np.float32)
    points[[5, 6]] = (cx - dx, hip - dy)
    points[[11, 12]] = (cx, hip)
    scores = np.full(17, 0.9, np.float32)
    width = aspect * height
    box = (cx - width / 2, hip - height * 0.6, cx + width / 2, hip + height * 0.4)
    return Track(track_id, box, 0.9, points, scores, identity, 10, 0)  # type: ignore[arg-type]


def frame(seq: int, tracks: list[Track]) -> FrameResult:
    t = seq / FPS
    record = FrameRecord(
        "cam1",
        seq,
        seq * 18000,
        Fraction(1, 90000),
        int(t * 1e9),
        datetime(2026, 1, 1, tzinfo=UTC),
        False,
        100,
        TimestampQuality.OK,
    )
    return FrameResult(record, tracks, analysed=True)


def standing(track_id: int = 0) -> Track:
    return make_track(track_id, HIP_UP, 5.0, 0.4)


def falling(track_id: int, k: float) -> Track:
    return make_track(
        track_id, HIP_UP + 120.0 * k, 5.0 + 80.0 * k, 0.4 + 2.0 * k, height=HEIGHT - 120.0 * k
    )


def fall_sequence(track_id: int = 0, recover: bool = True) -> list[list[Track]]:
    frames: list[list[Track]] = [[standing(track_id)] for _ in range(10)]
    frames += [[falling(track_id, 0.5)], [falling(track_id, 1.0)]]
    tail = standing(track_id) if recover else falling(track_id, 1.0)
    frames += [[tail] for _ in range(15)]
    return frames


def run(analyzer: FallAnalyzer, frames: list[list[Track]]) -> list:  # type: ignore[type-arg]
    events = []
    for seq, tracks in enumerate(frames):
        events += analyzer.update(frame(seq, tracks))
    return events


def test_fall_with_immediate_recovery_alerts_once() -> None:
    events = run(FallAnalyzer("cam1"), fall_sequence(recover=True))
    alerts = [e for e in events if not e.near_miss]
    assert len(alerts) == 1
    alert = alerts[0]
    assert alert.category is EventCategory.POSSIBLE_FALL
    assert alert.track_id == 0 and alert.confidence >= 0.5
    assert alert.start_s <= alert.end_s
    assert alert.analysed_seqs


def test_stumble_is_a_near_miss_and_never_an_alert() -> None:
    frames = [[standing()] for _ in range(10)]
    frames += [[make_track(0, HIP_UP + 30.0, 60.0, 0.6)]] * 2
    frames += [[standing()] for _ in range(10)]
    events = run(FallAnalyzer("cam1"), frames)
    assert events and all(e.near_miss for e in events)


def test_people_are_analysed_independently() -> None:
    frames = [[standing(1), t[0]] for t in fall_sequence(track_id=2)]
    events = [e for e in run(FallAnalyzer("cam1"), frames) if not e.near_miss]
    assert [e.track_id for e in events] == [2]


def test_uncertain_identity_still_alerts() -> None:
    frames = [
        [
            Track(
                t[0].track_id,
                t[0].box_xyxy,
                t[0].score,
                t[0].keypoints,
                t[0].keypoint_scores,
                "uncertain",
                1,
                0,
            )
        ]
        for t in fall_sequence()
    ]
    events = [e for e in run(FallAnalyzer("cam1"), frames) if not e.near_miss]
    assert len(events) == 1 and events[0].identity == "uncertain"


def test_track_lost_during_a_descent_raises_a_possible_fall() -> None:
    frames = [[standing()] for _ in range(10)]
    frames += [[falling(0, 0.5)]]  # suspected, then the person is no longer tracked
    frames += [[] for _ in range(10)]
    events = run(FallAnalyzer("cam1"), frames)
    lost = [e for e in events if e.evidence.get("track_lost")]
    assert len(lost) == 1
    assert lost[0].category is EventCategory.POSSIBLE_FALL


def test_frames_not_analysed_are_ignored() -> None:
    analyzer = FallAnalyzer("cam1")
    skipped = frame(0, [standing()])
    skipped = FrameResult(skipped.record, [], analysed=False)
    assert analyzer.update(skipped) == []


def test_confidence_formula_examples() -> None:
    strong = {
        "max_drop_heights": 0.6,
        "max_drop_speed_hps": 1.5,
        "max_torso_angle_deg": 85.0,
        "max_aspect": 2.4,
        "lying_s": 0.0,
    }
    assert fall_confidence(strong, confirmed=True) == pytest.approx(0.3 + 0.7 * 0.8071, abs=1e-3)
    weak = {
        "max_drop_heights": 0.15,
        "max_drop_speed_hps": 0.75,
        "max_torso_angle_deg": 60.0,
        "max_aspect": 0.6,
        "lying_s": 0.0,
    }
    assert fall_confidence(weak, confirmed=False) < 0.5
