import json
from datetime import UTC, datetime
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from fall_impact_monitor.analysis.floor import FloorAnalyzer
from fall_impact_monitor.analysis.zones import Zone, load_zones
from fall_impact_monitor.api.perception import DetectedPerson, FrameResult
from fall_impact_monitor.capture import FrameRecord, TimestampQuality
from fall_impact_monitor.events import EventCategory
from fall_impact_monitor.perception.tracking import Track

SIZE = (1000, 500)
LYING_BOX = (600.0, 400.0, 900.0, 480.0)  # wide, lower right
SOFA = Zone("sofa", "safe", ((0.0, 0.5), (0.4, 0.5), (0.4, 1.0), (0.0, 1.0)))
SOFA_BOX = (50.0, 300.0, 350.0, 380.0)


def lying_track(box: tuple[float, float, float, float] = LYING_BOX, track_id: int = 1) -> Track:
    points = np.zeros((17, 2), np.float32)
    points[[5, 6]] = (box[0] + 40, (box[1] + box[3]) / 2)
    points[[11, 12]] = (box[0] + 160, (box[1] + box[3]) / 2)
    return Track(track_id, box, 0.8, points, np.full(17, 0.9, np.float32), "stable", 10, 0)


def frame(
    t: float, tracks: list[Track], untracked: list[tuple[float, ...]] | None = None
) -> FrameResult:
    record = FrameRecord(
        "cam1",
        int(t),
        0,
        Fraction(1, 90000),
        int(t * 1e9),
        datetime(2026, 1, 1, tzinfo=UTC),
        False,
        1,
        TimestampQuality.OK,
    )
    people = [DetectedPerson(t_.box_xyxy, t_.score, None, t_.track_id) for t_ in tracks]
    people += [DetectedPerson(box, 0.15, None, None) for box in (untracked or [])]  # type: ignore[misc]
    return FrameResult(record, tracks, analysed=True, detections=people)


def feed(analyzer: FloorAnalyzer, seconds: int, **kwargs: object) -> list:  # type: ignore[type-arg]
    events = []
    for t in range(seconds):
        events += analyzer.update(frame(float(t), **kwargs), SIZE)  # type: ignore[arg-type]
    return events


def analyzer(zones: list[Zone] | None = None) -> FloorAnalyzer:
    return FloorAnalyzer("cam1", zones or [], person_down_s=30.0, no_fall_s=60.0)


def test_lying_outside_zones_without_an_observed_fall_fires_once_after_the_limit() -> None:
    events = feed(analyzer(), 125, tracks=[lying_track()])
    assert len(events) == 1
    event = events[0]
    assert event.category is EventCategory.PERSON_DOWN
    assert event.evidence["fall_observed"] is False
    assert event.track_id == 1 and not event.near_miss
    assert event.end_s - event.start_s >= 60.0


def test_lying_inside_a_safe_zone_never_fires() -> None:
    assert feed(analyzer([SOFA]), 125, tracks=[lying_track(SOFA_BOX)]) == []


def test_after_an_observed_fall_the_shorter_limit_applies() -> None:
    floor = analyzer()
    floor.note_fall(track_id=1, t_s=0.0)
    events = feed(floor, 40, tracks=[lying_track()])
    assert len(events) == 1 and events[0].evidence["fall_observed"] is True


def test_untracked_lying_detection_that_persists_fires() -> None:
    events = feed(analyzer(), 70, tracks=[], untracked=[LYING_BOX])
    assert len(events) == 1 and events[0].track_id is None


def test_getting_up_resets_the_episode() -> None:
    floor = analyzer()
    events = feed(floor, 50, tracks=[lying_track()])
    events += feed(floor, 5, tracks=[])  # gone / standing
    assert events == []


def test_upright_person_is_not_lying() -> None:
    upright = Track(1, (100.0, 100.0, 160.0, 400.0), 0.9, None, None, "stable", 10, 0)
    assert feed(analyzer(), 125, tracks=[upright]) == []


def test_zones_file_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "zones.json"
    path.write_text(
        json.dumps(
            {
                "zones": [
                    {
                        "name": "sofa",
                        "kind": "safe",
                        "polygon": [[0, 0.5], [0.4, 0.5], [0.4, 1], [0, 1]],
                    },
                    {
                        "name": "bed",
                        "kind": "safe",
                        "polygon": [[0.6, 0.5], [1, 0.5], [1, 1], [0.6, 1]],
                    },
                ]
            }
        )
    )
    zones = load_zones(path)
    assert [z.kind for z in zones] == ["safe", "safe"]
    assert zones[0].contains(0.2, 0.7) and not zones[0].contains(0.6, 0.7)


def test_invalid_zone_kind_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "zones.json"
    path.write_text(
        json.dumps({"zones": [{"name": "x", "kind": "bed", "polygon": [[0, 0], [1, 0], [1, 1]]}]})
    )
    with pytest.raises(ValueError, match="kind"):
        load_zones(path)


def test_small_lying_person_is_not_exempt() -> None:
    small = (700.0, 450.0, 800.0, 480.0)  # child-sized box, still lying on the floor
    assert len(feed(analyzer(), 65, tracks=[lying_track(small)])) == 1


def test_track_id_changes_do_not_restart_the_episode() -> None:
    floor = analyzer()
    events = []
    for t in range(70):
        tracks = [lying_track(track_id=1 + t // 20)]
        events += floor.update(frame(float(t), tracks), SIZE)
    assert len(events) == 1


def test_flicker_between_tracked_and_untracked_does_not_restart_the_episode() -> None:
    floor = analyzer()
    events = []
    for t in range(70):
        tracked = (t // 5) % 2 == 0
        tracks = [lying_track()] if tracked else []
        events += floor.update(frame(float(t), tracks, None if tracked else [LYING_BOX]), SIZE)
    assert len(events) == 1


def test_an_observed_fall_follows_the_person_across_track_ids() -> None:
    floor = analyzer()
    floor.note_fall(track_id=1, t_s=0.0)
    events = []
    for t in range(40):
        events += floor.update(frame(float(t), [lying_track(track_id=1 if t < 10 else 2)]), SIZE)
    assert len(events) == 1 and events[0].evidence["fall_observed"] is True


def test_a_single_non_lying_frame_does_not_restart_the_episode() -> None:
    floor = analyzer()
    upright = Track(1, (650.0, 200.0, 720.0, 480.0), 0.9, None, None, "stable", 10, 0)
    events = []
    for t in range(70):
        events += floor.update(frame(float(t), [upright if t % 10 == 5 else lying_track()]), SIZE)
    assert len(events) == 1


def test_a_threshold_above_the_initial_score_still_alerts_later() -> None:
    floor = FloorAnalyzer("cam1", [], person_down_s=30.0, no_fall_s=60.0, threshold=0.7)
    events = feed(floor, 125, tracks=[lying_track()])
    assert [e.near_miss for e in events] == [True, False]


@pytest.mark.parametrize("old_kind", ["allowed_lying", "hazard"])
def test_zone_kinds_replaced_by_safe_zones_are_rejected_with_a_hint(
    tmp_path: Path, old_kind: str
) -> None:
    path = tmp_path / "zones.json"
    zone = {"name": "x", "kind": old_kind, "polygon": [[0, 0], [1, 0], [1, 1]]}
    path.write_text(json.dumps({"zones": [zone]}))
    with pytest.raises(ValueError, match="safe"):
        load_zones(path)
