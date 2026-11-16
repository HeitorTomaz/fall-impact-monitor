"""The camera's mounting height in the person-on-the-floor rule (#78), on synthetic
people seen by a pinhole camera: a person lying head first towards a high camera keeps
a narrow box, but the shoulders are well below the hips."""

from fall_impact_monitor.analysis.floor import FloorAnalyzer
from fall_impact_monitor.events import EventCategory
from tests.analysis.pinhole import FPS, SIZE, fallen, scenario

LYING_TOWARD = scenario(lambda k: fallen(3.0, "toward", k), 2.5, 3.0, hold_s=70.0)


def down(frames: list, camera_height_m: float | None) -> list:  # type: ignore[type-arg]
    floor = FloorAnalyzer("cam1", [], camera_height_m=camera_height_m)
    events = [e for f in frames for e in floor.update(f, SIZE)]
    return [e for e in events if e.category is EventCategory.PERSON_DOWN]


def test_a_person_lying_head_first_towards_a_high_camera_is_on_the_floor() -> None:
    assert len(LYING_TOWARD) / FPS > 60.0  # beyond the limit without an observed fall
    assert down(LYING_TOWARD, 2.5)


def test_an_eye_level_or_unknown_camera_keeps_the_floor_rule_unchanged() -> None:
    """Head below the hips at eye level is someone bending; the rule waits a minute anyway,
    but the cue is only used for cameras mounted high."""
    assert down(LYING_TOWARD, 0.8) == down(LYING_TOWARD, None) == []
