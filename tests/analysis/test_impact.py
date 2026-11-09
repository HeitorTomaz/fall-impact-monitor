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
WALL = Zone("wall", "hazard", ((0.8, 0.0), (1.0, 0.0), (1.0, 1.0), (0.8, 1.0)))
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
    thumps: list[Thump] = (),
    falls: dict[int, list[Event]] | None = None,  # type: ignore[assignment]
    threshold: float = 0.5,
) -> list[Event]:
    analyzer = HeadImpactAnalyzer("cam1", [WALL], threshold=threshold)
    analyzer.add_thumps(list(thumps))
    events: list[Event] = []
    for i, x in enumerate(xs):
        tracks = [] if x is None else [track(x)]
        events += analyzer.update(result(i, tracks), SIZE, (falls or {}).get(i, []))
    return events


def into_wall() -> list[float | None]:
    # 1.5 person heights per second towards the wall, a stop inside it, then 4 s still.
    return [500.0 + 40.0 * i for i in range(8)] + [820.0] * 60


def fall(conf: float = 0.8, near: bool = False) -> Event:
    return make_event(
        camera_id="cam1",
        category=EventCategory.POSSIBLE_FALL,
        confidence=conf,
        start_s=100.0,
        end_s=100.6,
        track_id=1,
        identity="stable",
        evidence={"near_fall": True} if near else {},
        analysed_seqs=[],
        threshold=0.5,
    )


def impacts(events: list[Event]) -> list[Event]:
    return [e for e in events if e.category is EventCategory.POSSIBLE_HEAD_IMPACT]


def test_a_head_stop_at_a_wall_with_a_thump_is_an_alert() -> None:
    events = impacts(run(into_wall(), thumps=[Thump(100.55, 1.0)]))
    assert len(events) == 1 and not events[0].near_miss
    assert set(events[0].evidence["kinds"]) == {"head_stop_near_hazard", "thump"}
    assert events[0].confidence >= 0.5


def test_a_head_stop_alone_is_a_near_miss() -> None:
    events = impacts(run(into_wall()))
    assert len(events) == 1 and events[0].near_miss
    assert events[0].evidence["kinds"] == ["head_stop_near_hazard"]


def test_a_thump_with_nobody_near_a_hazard_is_nothing() -> None:
    assert impacts(run([300.0] * 60, thumps=[Thump(100.5, 1.0)])) == []


def test_two_weak_evidences_below_the_threshold_are_a_near_miss() -> None:
    events = impacts(run(into_wall(), thumps=[Thump(100.55, 0.1)], threshold=0.9))
    assert len(events) == 1 and events[0].near_miss
    assert len(events[0].evidence["kinds"]) == 2


def test_a_fall_with_the_head_in_a_hazard_zone_is_also_a_possible_head_impact() -> None:
    xs: list[float | None] = [850.0] * 60  # standing with the head inside the wall zone
    events = impacts(run(xs, falls={5: [fall()]}))
    assert len(events) == 1 and not events[0].near_miss
    assert set(events[0].evidence["kinds"]) == {"reaction", "head_in_hazard_zone"}


def test_evidence_outside_the_window_does_not_combine() -> None:
    events = impacts(run(into_wall(), thumps=[Thump(95.0, 1.0)]))
    assert len(events) == 1 and events[0].near_miss


def test_invalid_timestamps_never_give_kinematic_evidence() -> None:
    analyzer = HeadImpactAnalyzer("cam1", [WALL], threshold=0.5)
    events: list[Event] = []
    for i, x in enumerate(into_wall()):
        assert x is not None
        events += analyzer.update(result(i, [track(x)], TimestampQuality.ZERO_DELTA), SIZE, [])
    assert impacts(events) == []


def test_a_head_near_a_hazard_asks_for_full_rate_analysis() -> None:
    analyzer = HeadImpactAnalyzer("cam1", [WALL], threshold=0.5)
    analyzer.update(result(0, [track(300.0)]), SIZE, [])
    assert analyzer.attention_until() is None
    analyzer.update(result(1, [track(790.0)]), SIZE, [])
    until = analyzer.attention_until()
    assert until is not None and until > 100.0
