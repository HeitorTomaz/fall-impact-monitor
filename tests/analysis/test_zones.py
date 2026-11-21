import json
from pathlib import Path

import pytest

from fall_impact_monitor.analysis.zones import (
    Zone,
    load_zones,
    suggest_safe_zones,
    zone_at,
    zone_file_json,
)

LOW_CAMERA_M = 0.8
HIGH_CAMERA_M = 2.4


def test_a_detected_bed_becomes_a_suggested_zone_shrunk_towards_its_centre() -> None:
    zones = suggest_safe_zones(
        [("bed", (100.0, 200.0, 500.0, 600.0))], (1000, 1000), camera_height_m=HIGH_CAMERA_M
    )
    assert [(z.name, z.kind) for z in zones] == [("bed_1", "suggested_safe")]
    # High camera: the lowest 10 % of the box (front face) is dropped, then 10 % per side.
    expected = [0.14, 0.236, 0.46, 0.236, 0.46, 0.524, 0.14, 0.524]
    assert [v for point in zones[0].polygon for v in point] == pytest.approx(expected)


def test_a_suggested_zone_never_extends_beyond_the_detected_box() -> None:
    box = (101.0, 203.0, 507.0, 611.0)
    (zone,) = suggest_safe_zones([("sofa", box)], (1280, 720), camera_height_m=LOW_CAMERA_M)
    xs = [x * 1280 for x, _ in zone.polygon]
    ys = [y * 720 for _, y in zone.polygon]
    assert min(xs) >= box[0] and max(xs) <= box[2]
    assert min(ys) >= box[1] and max(ys) <= box[3]


def test_suggested_zones_are_numbered_per_label() -> None:
    found = [
        ("sofa", (0.0, 0.0, 10.0, 10.0)),
        ("bed", (20.0, 20.0, 40.0, 40.0)),
        ("sofa", (50.0, 50.0, 90.0, 90.0)),
    ]
    names = [z.name for z in suggest_safe_zones(found, (100, 100), camera_height_m=HIGH_CAMERA_M)]
    assert names == ["sofa_1", "bed_1", "sofa_2"]


def test_the_service_refuses_suggested_zones_until_they_are_checked(tmp_path: Path) -> None:
    zones = suggest_safe_zones(
        [("bed", (100.0, 200.0, 500.0, 600.0))], (1000, 1000), camera_height_m=HIGH_CAMERA_M
    )
    path = tmp_path / "zones.json"
    path.write_text(json.dumps(zone_file_json(zones, (1000, 1000))))
    with pytest.raises(ValueError, match="change its kind to 'safe'"):
        load_zones(path)
    assert load_zones(path, allow_suggested=True) == zones


def test_a_low_camera_drops_more_of_the_furniture_front_face_than_a_high_one() -> None:
    """The lower the camera, the more of the detected box is the furniture's front face,
    where nobody lies: 30 % at a low camera, 10 % at a high one, linear in between."""
    box = (0.0, 0.0, 100.0, 100.0)

    def bottom(height_m: float) -> float:
        (zone,) = suggest_safe_zones([("bed", box)], (100, 100), camera_height_m=height_m)
        return max(y for _, y in zone.polygon)

    assert bottom(LOW_CAMERA_M) == pytest.approx(0.70 - 0.07)  # 30 % cut, then 10 % shrink
    assert bottom(HIGH_CAMERA_M) == pytest.approx(0.90 - 0.09)
    assert bottom(1.6) == pytest.approx(0.80 - 0.08)  # halfway between 1.0 m and 2.2 m


BED_TOP = Zone("bed", "safe", ((0.2, 0.3), (0.8, 0.3), (0.8, 0.6), (0.2, 0.6)))


def test_a_person_lying_on_the_furniture_is_in_its_safe_zone() -> None:
    assert zone_at([BED_TOP], (300.0, 380.0, 700.0, 550.0), (1000, 1000)) == BED_TOP


def test_a_person_lying_on_the_floor_in_front_of_the_furniture_is_not_in_its_zone() -> None:
    """A camera mounted low sees the floor in front of a bed overlapping the bed: the centre
    of the person's box is over the bed, its bottom edge on the floor below it."""
    box = (300.0, 410.0, 700.0, 750.0)  # centre y 0.58 (over the bed), bottom 0.75
    assert BED_TOP.contains(0.5, 0.58)
    assert zone_at([BED_TOP], box, (1000, 1000)) is None
