from fall_impact_monitor.analysis.camera import CameraAnalyzer
from fall_impact_monitor.analysis.zones import Zone
from fall_impact_monitor.events import EventCategory
from tests.analysis.test_fall_analyzer import FPS, fall_sequence, falling, frame

SIZE = (1280, 720)


def run(analyzer: CameraAnalyzer, seconds_lying: float) -> list:  # type: ignore[type-arg]
    frames = fall_sequence(track_id=0, recover=False)
    frames += [[falling(0, 1.0)] for _ in range(int(seconds_lying * FPS))]
    events = []
    for seq, tracks in enumerate(frames):
        events += analyzer.update(frame(seq, tracks), SIZE)
    return events


def test_fall_then_lying_raises_a_fall_and_then_person_down_after_the_short_limit() -> None:
    analyzer = CameraAnalyzer("cam1", zones=[], person_down_s=30.0, no_fall_s=60.0)
    alerts = [e for e in run(analyzer, 35.0) if not e.near_miss]
    assert [e.category for e in alerts] == [EventCategory.POSSIBLE_FALL, EventCategory.PERSON_DOWN]
    assert alerts[1].evidence["fall_observed"] is True


def test_detectors_can_be_switched_off() -> None:
    analyzer = CameraAnalyzer("cam1", zones=[], fall_enabled=False, floor_enabled=False)
    assert run(analyzer, 35.0) == []


def test_lying_down_in_bed_raises_no_alert_from_either_detector() -> None:
    bed = Zone("bed", "safe", ((0.15, 0.3), (0.45, 0.3), (0.45, 0.6), (0.15, 0.6)))
    analyzer = CameraAnalyzer("cam1", zones=[bed], person_down_s=30.0, no_fall_s=60.0)
    events = run(analyzer, 130.0)
    assert events and not [e for e in events if not e.near_miss]


def test_head_impact_rules_run_without_zones_and_can_be_switched_off() -> None:
    from fall_impact_monitor.analysis.audio import Thump
    from tests.analysis.test_impact import fast_stop, result, track

    def impacts(enabled: bool) -> list[EventCategory]:
        analyzer = CameraAnalyzer("cam1", zones=[], head_impact_enabled=enabled)
        analyzer.add_thumps([Thump(100.55, 1.0)])
        found = []
        for i, x in enumerate(fast_stop()):
            assert x is not None
            found += [e.category for e in analyzer.update(result(i, [track(x)]), (1000, 1000))]
        return [c for c in found if c is EventCategory.POSSIBLE_HEAD_IMPACT]

    assert impacts(True) == [EventCategory.POSSIBLE_HEAD_IMPACT]
    assert impacts(False) == []


def test_audio_is_wanted_whenever_the_head_impact_rules_are_on() -> None:
    assert CameraAnalyzer("cam1", zones=[]).wants_audio()
    assert not CameraAnalyzer("cam1", zones=[], head_impact_enabled=False).wants_audio()


def test_flush_returns_the_head_impact_candidates_left_open() -> None:
    from tests.analysis.test_impact import fast_stop, result, track

    analyzer = CameraAnalyzer("cam1", zones=[])
    for i, x in enumerate(fast_stop()[:24]):
        assert x is not None
        analyzer.update(result(i, [track(x)]), (1000, 1000))
    assert [e.category for e in analyzer.flush()] == [EventCategory.POSSIBLE_HEAD_IMPACT]
    assert CameraAnalyzer("cam1", zones=[], head_impact_enabled=False).flush() == []
