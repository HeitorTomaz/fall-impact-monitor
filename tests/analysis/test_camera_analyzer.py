from fall_impact_monitor.analysis.camera import CameraAnalyzer
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
