import pytest

from fall_impact_monitor.events import Event, EventCategory, is_near_miss, make_event


def test_categories_match_the_architecture() -> None:
    assert {c.value for c in EventCategory} == {
        "possible_fall",
        "person_down",
        "possible_head_impact",
        "manual_report",
        "monitoring_degraded",
        "monitoring_restored",
    }


@pytest.mark.parametrize(
    ("confidence", "near_miss"),
    [(0.499999, True), (0.5, False), (0.500001, False)],
)
def test_threshold_boundary(confidence: float, near_miss: bool) -> None:
    assert is_near_miss(confidence, threshold=0.5) is near_miss


def test_make_event_clamps_confidence_and_sets_near_miss() -> None:
    event = make_event(
        camera_id="cam1",
        category=EventCategory.POSSIBLE_FALL,
        confidence=1.7,
        start_s=10.0,
        end_s=11.0,
        track_id=3,
        identity="stable",
        evidence={"drop_heights": 0.6},
        analysed_seqs=[1, 2, 3],
        threshold=0.5,
    )
    assert isinstance(event, Event)
    assert event.confidence == 1.0 and not event.near_miss
    assert len(event.event_id) == 36
    low = make_event(camera_id="cam1", category=EventCategory.PERSON_DOWN, confidence=-0.2,
                     start_s=0.0, end_s=1.0, track_id=None, identity="uncertain", evidence={},
                     analysed_seqs=[], threshold=0.5)
    assert low.confidence == 0.0 and low.near_miss


def test_event_window_must_be_ordered() -> None:
    with pytest.raises(ValueError):
        make_event(camera_id="c", category=EventCategory.POSSIBLE_FALL, confidence=0.9,
                   start_s=2.0, end_s=1.0, track_id=None, identity="stable", evidence={},
                   analysed_seqs=[], threshold=0.5)
