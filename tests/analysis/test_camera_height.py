"""The camera's mounting height in the fall rules (#78), on synthetic people seen by a
pinhole camera: a fall away from a high camera barely moves the hips down the image and
keeps the trunk upright in it, but the trunk and the box shrink."""

import pytest

from fall_impact_monitor.analysis.fall import FallAnalyzer
from fall_impact_monitor.analysis.fall_state import FallConfig
from fall_impact_monitor.events import EventCategory
from tests.analysis.pinhole import SIZE, fallen, scenario, sitting


def alerts(frames: list, camera_height_m: float | None) -> list:  # type: ignore[type-arg]
    analyzer = FallAnalyzer("cam1", FallConfig(camera_height_m=camera_height_m))
    events = [e for f in frames for e in analyzer.update(f, SIZE)]
    return [e for e in events if e.category is EventCategory.POSSIBLE_FALL and not e.near_miss]


@pytest.mark.parametrize("distance_m", [2.0, 3.5])
def test_a_fall_away_from_a_high_camera_is_raised(distance_m: float) -> None:
    frames = scenario(lambda k: fallen(distance_m, "away", k), 2.5, distance_m)
    assert alerts(frames, 2.5)


@pytest.mark.parametrize("distance_m", [2.0, 3.5])
@pytest.mark.parametrize("direction", ["sideways", "toward"])
def test_other_falls_under_a_high_camera_are_still_raised(
    direction: str, distance_m: float
) -> None:
    frames = scenario(lambda k: fallen(distance_m, direction, k), 2.5, distance_m)
    assert alerts(frames, 2.5)


@pytest.mark.parametrize("distance_m", [2.0, 3.5])
def test_sitting_down_under_a_high_camera_raises_no_alert(distance_m: float) -> None:
    """Sitting keeps the trunk long. (Crouching while facing a high camera shortens it
    like a fall away: a known limit of this cue, measured on the datasets.)"""
    frames = scenario(lambda k: sitting(distance_m, k), 2.5, distance_m)
    assert alerts(frames, 2.5) == []


@pytest.mark.parametrize("direction", ["away", "sideways", "toward"])
def test_an_eye_level_camera_keeps_the_rules_unchanged(direction: str) -> None:
    """The shortened-trunk cue means bending at eye level: it is never used there."""
    frames = scenario(lambda k: fallen(3.0, direction, k), 0.8, 3.0)
    with_height = [(e.start_s, e.confidence) for e in alerts(frames, 0.8)]
    without = [(e.start_s, e.confidence) for e in alerts(frames, None)]
    assert with_height == without


def test_a_shortened_trunk_counts_as_posture_evidence() -> None:
    from fall_impact_monitor.analysis.fall import fall_confidence

    upright = {
        "max_drop_heights": 0.15,
        "max_drop_speed_hps": 0.5,
        "max_torso_angle_deg": 0.0,
        "max_aspect": 0.6,
        "lying_s": 0.9,
    }
    shortened = {**upright, "min_torso_ratio": 0.3}
    assert fall_confidence(shortened, rule_met=True) > fall_confidence(upright, rule_met=True)
    assert fall_confidence({**upright, "min_torso_ratio": 0.7}, rule_met=True) == (
        fall_confidence(upright, rule_met=True)
    )


def test_a_person_lying_head_first_towards_a_high_camera_is_on_the_floor() -> None:
    """The box stays narrow along the line of sight, but the shoulders are well below the
    hips: under a high camera that is lying (it lasts seconds when bending, the floor rule
    waits a minute)."""
    from fall_impact_monitor.analysis.floor import FloorAnalyzer
    from tests.analysis.pinhole import FPS

    frames = scenario(lambda k: fallen(3.0, "toward", k), 2.5, 3.0, hold_s=70.0)

    def down(camera_height_m: float | None) -> list:  # type: ignore[type-arg]
        floor = FloorAnalyzer("cam1", [], camera_height_m=camera_height_m)
        events = [e for f in frames for e in floor.update(f, SIZE)]
        return [e for e in events if e.category is EventCategory.PERSON_DOWN]

    assert len(frames) / FPS > 60.0
    assert down(2.5)
    assert down(0.8) == down(None) == []
