import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from fall_impact_monitor.cli.main import main
from tests.conftest import FakePlayback
from tests.recording.media import write_camera_clip


def test_clip_command_exports_and_prints_both_windows(
    playback: FakePlayback, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    clip = write_camera_clip(tmp_path / "camera.mov", 10.0).read_bytes()
    listing = [{"start": (start - timedelta(minutes=1)).isoformat(), "duration": 600.0}]
    playback.routes["/list"] = lambda q: (200, json.dumps(listing).encode())
    playback.routes["/get"] = lambda q: (200, clip)
    out = tmp_path / "clip.mp4"
    code = main(
        [
            "clip",
            "--camera",
            "cam1",
            "--start",
            "2026-01-01T00:00:00Z",
            "--duration-s",
            "10",
            "--out",
            str(out),
            "--api-url",
            playback.url,
        ]
    )
    assert code == 0
    assert out.exists() and out.with_suffix(".json").exists()
    printed = capsys.readouterr().out
    assert "requested 2026-01-01T00:00:00+00:00" in printed
    assert "actual" in printed
