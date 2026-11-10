import dataclasses
import math
from datetime import UTC, datetime
from fractions import Fraction

import numpy as np
import pytest

from fall_impact_monitor.analysis.fall import FallAnalyzer, fall_confidence
from fall_impact_monitor.analysis.zones import Zone
from fall_impact_monitor.api.perception import FrameResult
from fall_impact_monitor.capture import FrameRecord, TimestampQuality
from fall_impact_monitor.events import EventCategory
from fall_impact_monitor.perception.tracking import Track

FPS = 5.0
SIZE = (1280, 720)
HEIGHT = 200.0
HIP_UP = 150.0


def make_track(
    track_id: int,
    hip: float,
    angle: float,
    aspect: float,
    height: float = HEIGHT,
    identity: str = "stable",
    cx: float | None = None,
) -> Track:
    torso = 60.0
    dx, dy = torso * math.sin(math.radians(angle)), torso * math.cos(math.radians(angle))
    cx = 300.0 + 200.0 * track_id if cx is None else cx
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
        events += analyzer.update(frame(seq, tracks), SIZE)
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
    assert analyzer.update(skipped, SIZE) == []


def test_confidence_formula_examples() -> None:
    strong = {
        "max_drop_heights": 0.6,
        "max_drop_speed_hps": 1.5,
        "max_torso_angle_deg": 85.0,
        "max_aspect": 2.4,
        "lying_s": 0.0,
    }
    assert fall_confidence(strong, rule_met=True) == pytest.approx(0.3 + 0.7 * 0.8071, abs=1e-3)
    weak = {
        "max_drop_heights": 0.15,
        "max_drop_speed_hps": 0.75,
        "max_torso_angle_deg": 60.0,
        "max_aspect": 0.6,
        "lying_s": 0.0,
    }
    assert fall_confidence(weak, rule_met=False) < 0.5


def test_a_new_track_id_after_a_brief_loss_keeps_the_person_history() -> None:
    frames = [[standing()] for _ in range(10)]
    frames += [[] for _ in range(3)]  # the tracker loses the person during the fall
    lying = dataclasses.replace(falling(0, 1.0), track_id=5)
    frames += [[lying] for _ in range(10)]
    alerts = [e for e in run(FallAnalyzer("cam1"), frames) if not e.near_miss]
    assert [(e.category, e.track_id) for e in alerts] == [(EventCategory.POSSIBLE_FALL, 5)]


def test_an_unconfirmed_suspicion_above_the_threshold_is_notified() -> None:
    frames = [[standing()] for _ in range(10)]
    frames += [[make_track(0, HIP_UP + 30.0, 60.0, 0.6)]] * 2
    frames += [[standing()] for _ in range(10)]
    events = run(FallAnalyzer("cam1", threshold=0.1), frames)
    assert events and not any(e.near_miss for e in events)


def test_a_track_lost_during_a_weak_descent_is_still_an_alert() -> None:
    frames = [[standing()] for _ in range(10)]
    frames += [[make_track(0, HIP_UP + 30.0, 56.0, 0.6)]]
    frames += [[] for _ in range(10)]
    lost = [e for e in run(FallAnalyzer("cam1"), frames) if e.evidence.get("track_lost")]
    assert len(lost) == 1 and not lost[0].near_miss


# A bed seen from the side: lying in it puts the box centre at about (0.23, 0.36).
BED = Zone("bed", "safe", ((0.15, 0.3), (0.45, 0.3), (0.45, 0.6), (0.15, 0.6)))


def in_bed() -> Track:
    return falling(0, 1.0)


def on_floor(cx: float = 300.0, hip: float = 480.0) -> Track:
    return make_track(0, hip, 85.0, 2.2, height=80.0, cx=cx)


def lie_down_in_bed(seconds: float) -> list[list[Track]]:
    frames = fall_sequence(recover=False)
    return frames + [[in_bed()] for _ in range(int(seconds * FPS))]


def test_lying_down_quickly_in_a_safe_zone_is_a_near_miss_only() -> None:
    events = run(FallAnalyzer("cam1", zones=[BED]), lie_down_in_bed(10.0))
    falls = [e for e in events if e.category is EventCategory.POSSIBLE_FALL]
    assert falls and all(e.near_miss for e in falls)
    assert any(e.evidence.get("safe_zone") == "bed" for e in falls)


def test_the_same_descent_outside_any_safe_zone_alerts() -> None:
    elsewhere = Zone("sofa", "safe", ((0.7, 0.3), (0.9, 0.3), (0.9, 0.6), (0.7, 0.6)))
    events = run(FallAnalyzer("cam1", zones=[elsewhere]), lie_down_in_bed(10.0))
    assert [e for e in events if not e.near_miss]


def test_falling_from_the_bed_to_the_floor_alerts() -> None:
    frames = lie_down_in_bed(10.0)
    frames += [[on_floor()] for _ in range(10)]
    alerts = [e for e in run(FallAnalyzer("cam1", zones=[BED]), frames) if not e.near_miss]
    assert len(alerts) == 1
    assert alerts[0].category is EventCategory.POSSIBLE_FALL


def test_rolling_out_of_the_bed_sideways_without_a_drop_alerts() -> None:
    frames = lie_down_in_bed(2.0)
    frames += [[on_floor(cx=100.0, hip=270.0)] for _ in range(10)]
    alerts = [e for e in run(FallAnalyzer("cam1", zones=[BED]), frames) if not e.near_miss]
    assert len(alerts) == 1
    assert alerts[0].evidence.get("left_safe_zone") == "bed"


def test_a_single_frame_outside_the_bed_edge_does_not_alert() -> None:
    frames = lie_down_in_bed(10.0)
    for _ in range(5):
        frames += [[on_floor(cx=100.0, hip=270.0)]] + [[in_bed()] for _ in range(5)]
    events = run(FallAnalyzer("cam1", zones=[BED]), frames)
    assert not [e for e in events if not e.near_miss]


def test_getting_out_of_bed_and_walking_away_does_not_alert() -> None:
    frames = lie_down_in_bed(10.0)
    frames += [[standing()] for _ in range(20)]
    frames += [[make_track(0, HIP_UP, 5.0, 0.4, cx=100.0)] for _ in range(20)]
    events = run(FallAnalyzer("cam1", zones=[BED]), frames)
    assert not [e for e in events if not e.near_miss]


def test_a_track_lost_while_falling_into_a_safe_zone_is_a_near_miss() -> None:
    zone = Zone("bed", "safe", ((0.15, 0.2), (0.45, 0.2), (0.45, 0.6), (0.15, 0.6)))
    frames = [[standing()] for _ in range(10)]
    frames += [[falling(0, 0.5)]]  # box centre at about (0.23, 0.27)
    frames += [[] for _ in range(10)]
    lost = [
        e for e in run(FallAnalyzer("cam1", zones=[zone]), frames) if e.evidence.get("track_lost")
    ]
    assert len(lost) == 1 and lost[0].near_miss
    assert lost[0].evidence.get("safe_zone") == "bed"
