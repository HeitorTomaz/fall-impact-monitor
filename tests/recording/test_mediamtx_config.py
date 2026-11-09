import json
import os
from pathlib import Path

import pytest

from fall_impact_monitor.cli.main import main
from fall_impact_monitor.config import CameraSettings
from fall_impact_monitor.recording.mediamtx_config import (
    PUBLISHER,
    build_mediamtx_config,
    write_mediamtx_config,
)


def test_each_camera_gets_a_recorded_main_path_and_a_relayed_substream() -> None:
    cameras = [
        CameraSettings("cam1", "rtsp://u:p@h/1", "rtsp://u:p@h/2"),
        CameraSettings("cam2", "rtsp://u:p@h/3", None),
    ]
    config = build_mediamtx_config(cameras, max_age_h=168)
    assert config["paths"]["cam1"] == {"source": "rtsp://u:p@h/1", "record": True}
    assert config["paths"]["cam1sub"] == {"source": "rtsp://u:p@h/2", "record": False}
    assert config["paths"]["cam2"] == {"source": "rtsp://u:p@h/3", "record": True}
    assert "cam2sub" not in config["paths"]
    assert config["pathDefaults"]["recordDeleteAfter"] == "168h"
    assert config["pathDefaults"]["recordFormat"] == "fmp4"
    assert config["playback"] is True
    assert config["api"] is False


def test_publisher_value_accepts_a_simulated_camera() -> None:
    config = build_mediamtx_config([CameraSettings("cam1", PUBLISHER, None)], max_age_h=1.5)
    assert config["paths"]["cam1"] == {"source": "publisher", "record": True}
    assert config["pathDefaults"]["recordDeleteAfter"] == "1.5h"


def test_written_config_is_json_readable_only_by_its_owner(tmp_path: Path) -> None:
    config = build_mediamtx_config([CameraSettings("cam1", "rtsp://h/1", None)], max_age_h=24)
    out = tmp_path / "conf" / "mediamtx.json"
    write_mediamtx_config(config, out)
    assert json.loads(out.read_text(encoding="utf-8")) == config
    if os.name == "posix":
        assert out.stat().st_mode & 0o777 == 0o600


def test_cli_writes_the_config_and_never_prints_camera_urls(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FIM_CAMERAS", "cam1")
    monkeypatch.setenv("FIM_CAMERA_CAM1_MAIN_URL", "rtsp://user:secret@host/1")
    out = tmp_path / "mediamtx.json"
    assert main(["mediamtx-config", "--out", str(out)]) == 0
    assert "cam1" in json.loads(out.read_text(encoding="utf-8"))["paths"]
    captured = capsys.readouterr()
    assert "secret" not in captured.out + captured.err
