import json
from pathlib import Path

import pytest

from fall_impact_monitor.analysis.zones import load_zones, suggest_safe_zones, zone_file_json


def test_a_detected_bed_becomes_a_safe_zone_shrunk_towards_its_centre() -> None:
    zones = suggest_safe_zones([("bed", (100.0, 200.0, 500.0, 600.0))], (1000, 1000))
    assert [(z.name, z.kind) for z in zones] == [("bed_1", "safe")]
    expected = [0.14, 0.24, 0.46, 0.24, 0.46, 0.56, 0.14, 0.56]
    assert [v for point in zones[0].polygon for v in point] == pytest.approx(expected)


def test_suggested_zones_are_numbered_per_label() -> None:
    found = [
        ("sofa", (0.0, 0.0, 10.0, 10.0)),
        ("bed", (20.0, 20.0, 40.0, 40.0)),
        ("sofa", (50.0, 50.0, 90.0, 90.0)),
    ]
    names = [z.name for z in suggest_safe_zones(found, (100, 100))]
    assert names == ["sofa_1", "bed_1", "sofa_2"]


def test_a_zone_file_written_from_zones_loads_back_the_same_zones(tmp_path: Path) -> None:
    zones = suggest_safe_zones([("bed", (100.0, 200.0, 500.0, 600.0))], (1000, 1000))
    path = tmp_path / "zones.json"
    path.write_text(json.dumps(zone_file_json(zones, (1000, 1000))))
    assert load_zones(path) == zones
