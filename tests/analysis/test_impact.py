from datetime import UTC, datetime
from fractions import Fraction

import numpy as np

from fall_impact_monitor.analysis.audio import Thump
from fall_impact_monitor.analysis.impact import HeadImpactAnalyzer
from fall_impact_monitor.analysis.zones import Zone
from fall_impact_monitor.api.perception import FrameResult
from fall_impact_monitor.capture import FrameRecord, TimestampQuality
from fall_impact_monitor.events import Event, EventCategory, make_event
from fall_impact_monitor.perception.tracking import Track

SIZE = (1000, 1000)
# A soft place (bed, sofa) where the fast stop of ``fast_stop()`` ends.
SAFE = Zone("bed", "safe", ((0.8, 0.0), (1.0, 0.0), (1.0, 1.0), (0.8, 1.0)))
FPS = 15.0


def track(x: float, track_id: int = 1) -> Track:
    points = np.zeros((17, 2), np.float32)
    scores = np.zeros(17, np.float32)
    points[0], scores[0] = (x, 320.0), 0.9
    box = (x - 50.0, 300.0, x + 50.0, 700.0)  # 400 px tall
    return Track(track_id, box, 0.9, points, scores, "stable", 10, 0)  # type: ignore[arg-type]


def result(
    i: int, tracks: list[Track], quality: TimestampQuality = TimestampQuality.OK
) -> FrameResult:
    t = 100.0 + i / FPS
    record = FrameRecord(
        "cam1",
        i,
        i,
        Fraction(1, 90000),
        int(t * 1e9),
        datetime(2026, 11, 9, tzinfo=UTC),
        False,
        100,
        quality,
    )
    return FrameResult(record, tracks, analysed=True)


def run(
    xs: list[float | None],
    *,
    zones: list[Zone] | None = None,
    thumps: list[Thump] = (),  # type: ignore[assignment]
    falls: dict[int, list[Event]] | None = None,
    threshold: float = 0.5,
) -> list[Event]:
    analyzer = HeadImpactAnalyzer("cam1", zones or [], threshold=threshold)
    analyzer.add_thumps(list(thumps))
    events: list[Event] = []
    for i, x in enumerate(xs):
        tracks = [] if x is None else [track(x)]
        events += analyzer.update(result(i, tracks), SIZE, (falls or {}).get(i, []))
    return events


def fast_stop() -> list[float | None]:
    # 1.5 person heights per second, a sudden stop at x = 820, then 4 s still.
    return [500.0 + 40.0 * i for i in range(8)] + [820.0] * 120


def fall(
    conf: float = 0.8, near: bool = False, start_s: float = 100.0, **evidence: object
) -> Event:
    return make_event(
        camera_id="cam1",
        category=EventCategory.POSSIBLE_FALL,
        confidence=conf,
        start_s=start_s,
        end_s=start_s + 0.6,
        track_id=1,
        identity="stable",
        evidence={**({"near_fall": True} if near else {}), **evidence},
        analysed_seqs=[],
        threshold=0.5,
    )


def impacts(events: list[Event]) -> list[Event]:
    return [e for e in events if e.category is EventCategory.POSSIBLE_HEAD_IMPACT]


def test_a_fast_head_stop_with_a_thump_is_an_alert_without_any_zone() -> None:
    events = impacts(run(fast_stop(), thumps=[Thump(100.55, 1.0)]))
    assert len(events) == 1 and not events[0].near_miss
    assert set(events[0].evidence["kinds"]) == {"head_stop", "thump"}
    assert events[0].confidence >= 0.5


def test_a_head_stop_alone_is_a_near_miss() -> None:
    events = impacts(run(fast_stop()))
    assert len(events) == 1 and events[0].near_miss
    assert events[0].evidence["kinds"] == ["head_stop"]


def test_a_head_stop_inside_a_safe_zone_is_not_evidence() -> None:
    assert impacts(run(fast_stop(), zones=[SAFE], thumps=[Thump(100.55, 1.0)])) == []


def test_a_thump_with_nobody_moving_is_nothing() -> None:
    assert impacts(run([300.0] * 120, thumps=[Thump(100.5, 1.0)])) == []


def test_two_weak_evidences_below_the_threshold_are_a_near_miss() -> None:
    events = impacts(run(fast_stop(), thumps=[Thump(100.55, 0.1)], threshold=0.9))
    assert len(events) == 1 and events[0].near_miss
    assert len(events[0].evidence["kinds"]) == 2


def test_a_fall_with_a_fast_head_stop_is_also_a_possible_head_impact() -> None:
    events = impacts(run(fast_stop(), falls={8: [fall(start_s=100.3)]}))
    assert len(events) == 1 and not events[0].near_miss
    assert set(events[0].evidence["kinds"]) == {"head_stop", "reaction"}


def test_a_fall_without_a_head_stop_is_only_a_near_miss_here() -> None:
    events = impacts(run([500.0] * 120, falls={5: [fall()]}))
    assert len(events) == 1 and events[0].near_miss
    assert events[0].evidence["kinds"] == ["reaction"]


def test_evidence_outside_the_window_does_not_combine() -> None:
    events = impacts(run(fast_stop(), thumps=[Thump(95.0, 1.0)]))
    assert len(events) == 1 and events[0].near_miss


def test_invalid_timestamps_never_give_kinematic_evidence() -> None:
    analyzer = HeadImpactAnalyzer("cam1", [], threshold=0.5)
    events: list[Event] = []
    for i, x in enumerate(fast_stop()):
        assert x is not None
        events += analyzer.update(result(i, [track(x)], TimestampQuality.ZERO_DELTA), SIZE, [])
    assert impacts(events) == []


def test_a_fast_head_asks_for_full_rate_analysis() -> None:
    analyzer = HeadImpactAnalyzer("cam1", [], threshold=0.5)
    for i in range(8):
        analyzer.update(result(i, [track(300.0 + 40.0 * i)]), SIZE, [])
    until = analyzer.attention_until()
    assert until is not None and until > 100.0


def test_walking_does_not_ask_for_full_rate_analysis() -> None:
    # About 0.6 person heights per second: a person walking across the room.
    analyzer = HeadImpactAnalyzer("cam1", [], threshold=0.5)
    for i in range(30):
        analyzer.update(result(i, [track(100.0 + 16.0 * i)]), SIZE, [])
    assert analyzer.attention_until() is None


def test_a_fall_confirmed_seconds_later_joins_the_head_stop() -> None:
    # The fall started with the head stop (100.6 s) but was confirmed 4 s later.
    events = impacts(run(fast_stop(), falls={66: [fall(start_s=100.6)]}))
    assert len(events) == 1
    assert {"head_stop", "reaction"} <= set(events[0].evidence["kinds"])


def test_lying_down_quickly_in_a_safe_zone_is_not_a_head_impact() -> None:
    in_bed = fall(start_s=100.3, safe_zone="bed")  # stored as a near-miss by the fall rule
    events = impacts(run(fast_stop(), zones=[SAFE], falls={8: [in_bed]}))
    assert all(e.near_miss for e in events)
    assert all("head_stop" not in e.evidence["kinds"] for e in events)


def test_people_who_left_are_forgotten() -> None:
    analyzer = HeadImpactAnalyzer("cam1", [], threshold=0.5)
    analyzer.update(result(0, [track(300.0, track_id=7)]), SIZE, [])
    assert analyzer.people_count() == 1
    analyzer.update(result(int(31 * FPS), []), SIZE, [])
    assert analyzer.people_count() == 0


def test_events_list_the_analysed_frames_of_their_evidence() -> None:
    events = impacts(run(fast_stop(), thumps=[Thump(100.55, 1.0)]))
    assert events[0].analysed_seqs
