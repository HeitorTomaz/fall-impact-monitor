from pathlib import Path

import numpy as np
import pytest

from fall_impact_monitor.capture.audio import AudioReader
from tests.recording.media import write_camera_clip


def collect(path: Path) -> list[tuple[float, np.ndarray]]:
    chunks: list[tuple[float, np.ndarray]] = []
    reader = AudioReader(str(path), "cam1", lambda t, s: chunks.append((t, s)))
    reader.run()  # a file ends: run() returns
    return chunks


def test_camera_audio_is_delivered_as_8khz_mono_with_increasing_times(tmp_path: Path) -> None:
    chunks = collect(write_camera_clip(tmp_path / "camera.mov", 3.0))
    total = sum(len(s) for _, s in chunks)
    assert abs(total - 3 * 8000) <= 800
    assert all(s.dtype == np.float32 and s.ndim == 1 for _, s in chunks)
    times = [t for t, _ in chunks]
    assert times == sorted(times)


def test_a_source_without_audio_stops_cleanly(tmp_path: Path) -> None:
    assert collect(write_camera_clip(tmp_path / "silent.mov", 1.0, audio=False)) == []


def test_the_relay_is_asked_for_the_audio_track_only(monkeypatch: pytest.MonkeyPatch) -> None:
    from fall_impact_monitor.capture import audio

    seen: dict[str, str] = {}

    def fake_open(url: str, options: dict[str, str], timeout: object) -> None:
        seen.update(options)
        reader.stop()
        raise OSError("relay down")

    monkeypatch.setattr(audio.av, "open", fake_open)
    reader = AudioReader("rtsp://relay:8554/cam1", "cam1", lambda t, s: None, sleep=lambda s: None)
    reader.run()
    assert seen == {"rtsp_transport": "tcp", "allowed_media_types": "audio"}
