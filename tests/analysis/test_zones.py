import json
from pathlib import Path

import pytest

from fall_impact_monitor.analysis.zones import load_zones, suggest_safe_zones, zone_file_json


def test_a_detected_bed_becomes_a_suggested_zone_shrunk_towards_its_centre() -> None:
    zones = suggest_safe_zones([("bed", (100.0, 200.0, 500.0, 600.0))], (1000, 1000))
    assert [(z.name, z.kind) for z in zones] == [("bed_1", "suggested_safe")]
    expected = [0.14, 0.24, 0.46, 0.24, 0.46, 0.56, 0.14, 0.56]
    assert [v for point in zones[0].polygon for v in point] == pytest.approx(expected)


def test_a_suggested_zone_never_extends_beyond_the_detected_box() -> None:
    box = (101.0, 203.0, 507.0, 611.0)
    (zone,) = suggest_safe_zones([("sofa", box)], (1280, 720))
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
    names = [z.name for z in suggest_safe_zones(found, (100, 100))]
    assert names == ["sofa_1", "bed_1", "sofa_2"]


def test_the_service_refuses_suggested_zones_until_they_are_checked(tmp_path: Path) -> None:
    zones = suggest_safe_zones([("bed", (100.0, 200.0, 500.0, 600.0))], (1000, 1000))
    path = tmp_path / "zones.json"
    path.write_text(json.dumps(zone_file_json(zones, (1000, 1000))))
    with pytest.raises(ValueError, match="change its kind to 'safe'"):
        load_zones(path)
    assert load_zones(path, allow_suggested=True) == zones
