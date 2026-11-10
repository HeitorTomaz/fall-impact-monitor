import json
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from fall_impact_monitor.analysis.zones import load_zones
from fall_impact_monitor.cli import zones as zones_cli
from fall_impact_monitor.cli.main import main
from fall_impact_monitor.perception.tracking import Box


def zones_cmd(source: Path, out: Path, models_dir: Path) -> list[str]:
    return [
        "zones",
        "--source",
        str(source),
        "--out-dir",
        str(out),
        "--models-dir",
        str(models_dir),
    ]


def fake_detector(image: NDArray[np.uint8]) -> list[tuple[str, Box]]:
    return [("bed", (64.0, 36.0, 320.0, 180.0)), ("sofa", (400.0, 200.0, 600.0, 300.0))]


def test_zones_without_the_model_writes_a_snapshot_and_no_zones(
    synthetic_video: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "zones"
    assert main(zones_cmd(synthetic_video, out, tmp_path / "no-models")) == 0
    assert (out / "snapshot.png").stat().st_size > 0
    template = json.loads((out / "zones.json").read_text())
    assert template["frame_size"] == [640, 360]
    assert load_zones(out / "zones.json") == []
    printed = capsys.readouterr().out
    assert "FIM_MODELS_FETCH" in printed
    assert printed.count("fim models fetch") == 1


def test_zones_writes_detected_beds_and_sofas_as_suggestions(
    synthetic_video: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(zones_cli, "furniture_detector", lambda args: fake_detector)
    out = tmp_path / "zones"
    assert main(zones_cmd(synthetic_video, out, tmp_path)) == 0
    zones = load_zones(out / "zones.json", allow_suggested=True)
    assert [(z.name, z.kind) for z in zones] == [
        ("bed_1", "suggested_safe"),
        ("sofa_1", "suggested_safe"),
    ]
    assert (out / "zones.png").stat().st_size > 0


def test_check_draws_suggestions_but_fails_until_they_are_accepted(
    synthetic_video: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(zones_cli, "furniture_detector", lambda args: fake_detector)
    assert main(zones_cmd(synthetic_video, tmp_path, tmp_path)) == 0
    (tmp_path / "zones.png").unlink()
    assert main(["zones", "--check", str(tmp_path / "zones.json")]) == 1
    assert (tmp_path / "zones.png").stat().st_size > 0
    assert "not used until" in capsys.readouterr().out
    accepted = (tmp_path / "zones.json").read_text().replace("suggested_safe", "safe")
    (tmp_path / "zones.json").write_text(accepted)
    assert main(["zones", "--check", str(tmp_path / "zones.json")]) == 0


def test_zones_does_not_overwrite_an_existing_zone_file(
    synthetic_video: Path, tmp_path: Path
) -> None:
    (tmp_path / "zones.json").write_text('{"zones": []}')
    assert main(zones_cmd(synthetic_video, tmp_path, tmp_path)) == 0
    assert (tmp_path / "zones.json").read_text() == '{"zones": []}'


def test_zones_reports_an_existing_invalid_zone_file(synthetic_video: Path, tmp_path: Path) -> None:
    old = {"name": "wall", "kind": "hazard", "polygon": [[0, 0], [1, 0], [1, 1]]}
    (tmp_path / "zones.json").write_text(json.dumps({"zones": [old]}))
    assert main(zones_cmd(synthetic_video, tmp_path, tmp_path)) == 1


def test_zones_check_validates_a_file(tmp_path: Path) -> None:
    bad = tmp_path / "zones.json"
    bad.write_text('{"zones": [{"name": "x", "kind": "bed", "polygon": [[0, 0], [1, 0], [1, 1]]}]}')
    assert main(["zones", "--check", str(bad)]) == 1


def test_zones_check_redraws_the_overview_next_to_the_snapshot(
    synthetic_video: Path, tmp_path: Path
) -> None:
    assert main(zones_cmd(synthetic_video, tmp_path, tmp_path / "no-models")) == 0
    (tmp_path / "zones.png").unlink()
    zone = {"name": "bed", "kind": "safe", "polygon": [[0.1, 0.1], [0.5, 0.1], [0.5, 0.5]]}
    (tmp_path / "zones.json").write_text(json.dumps({"zones": [zone]}))
    assert main(["zones", "--check", str(tmp_path / "zones.json")]) == 0
    assert (tmp_path / "zones.png").stat().st_size > 0


def test_zones_says_how_to_fix_a_folder_it_cannot_write(
    synthetic_video: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def refuse(self: Path, *args: object, **kwargs: object) -> None:
        raise PermissionError(13, "Permission denied", str(self))

    monkeypatch.setattr(Path, "mkdir", refuse)
    assert main(zones_cmd(synthetic_video, tmp_path / "zones", tmp_path)) == 1
    assert "chown" in capsys.readouterr().out


def test_zones_check_reports_an_overview_it_cannot_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import cv2

    cv2.imwrite(str(tmp_path / "snapshot.png"), np.zeros((36, 64, 3), np.uint8))
    zone = {"name": "bed", "kind": "safe", "polygon": [[0.1, 0.1], [0.5, 0.1], [0.5, 0.5]]}
    (tmp_path / "zones.json").write_text(json.dumps({"zones": [zone]}))
    monkeypatch.setattr(zones_cli.cv2, "imwrite", lambda *args: False)
    assert main(["zones", "--check", str(tmp_path / "zones.json")]) == 1
    assert "could not write" in capsys.readouterr().out
