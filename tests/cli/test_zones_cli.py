import json
from pathlib import Path

from fall_impact_monitor.analysis.zones import load_zones
from fall_impact_monitor.cli.main import main


def test_zones_writes_a_snapshot_and_a_loadable_template(
    synthetic_video: Path, tmp_path: Path
) -> None:
    out = tmp_path / "zones"
    assert main(["zones", "--source", str(synthetic_video), "--out-dir", str(out)]) == 0
    assert (out / "snapshot.png").stat().st_size > 0
    template = json.loads((out / "zones.json").read_text())
    assert template["frame_size"] == [640, 360]
    assert {z.kind for z in load_zones(out / "zones.json")} == {"safe"}


def test_zones_does_not_overwrite_an_existing_zone_file(
    synthetic_video: Path, tmp_path: Path
) -> None:
    (tmp_path / "zones.json").write_text('{"zones": []}')
    assert main(["zones", "--source", str(synthetic_video), "--out-dir", str(tmp_path)]) == 0
    assert (tmp_path / "zones.json").read_text() == '{"zones": []}'


def test_zones_check_validates_a_file(tmp_path: Path) -> None:
    bad = tmp_path / "zones.json"
    bad.write_text('{"zones": [{"name": "x", "kind": "bed", "polygon": [[0, 0], [1, 0], [1, 1]]}]}')
    assert main(["zones", "--check", str(bad)]) == 1
