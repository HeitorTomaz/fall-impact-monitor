"""End to end against a real MediaMTX (no camera): the generated configuration is
accepted, a published stream with G711 audio is recorded, a reconnect leaves a visible
gap, and a clip over the gap is exported with AAC audio and the gap listed.

Runs only when FIM_TEST_MEDIAMTX_RTSP and FIM_TEST_MEDIAMTX_PLAYBACK are set: the
`mediamtx-test` service (compose profile `itest`) or the CI step that starts MediaMTX
with ``tests/integration/mediamtx.json``.
"""

import json
import os
import time
from datetime import UTC, datetime, timedelta
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import pytest

from fall_impact_monitor.config import CameraSettings
from fall_impact_monitor.recording import export_clip, list_spans
from fall_impact_monitor.recording.clip_media import probe
from fall_impact_monitor.recording.mediamtx_config import PUBLISHER, build_mediamtx_config

FIXTURE = Path(__file__).with_name("mediamtx.json")
RTSP = os.environ.get("FIM_TEST_MEDIAMTX_RTSP", "")
PLAYBACK = os.environ.get("FIM_TEST_MEDIAMTX_PLAYBACK", "")
PATH = "itest"


def test_the_fixture_is_the_generated_configuration() -> None:
    expected = build_mediamtx_config([CameraSettings(PATH, PUBLISHER, None)], max_age_h=1)
    assert json.loads(FIXTURE.read_text(encoding="utf-8")) == expected


def publish(seconds: float) -> None:
    """Publish a moving bar with a tone, in real time, like a camera."""
    fps, width, height = 15, 320, 240
    with av.open(f"{RTSP}/{PATH}", "w", format="rtsp", options={"rtsp_transport": "tcp"}) as out:
        video = out.add_stream("libx264", rate=fps)
        video.width, video.height, video.pix_fmt = width, height, "yuv420p"
        video.options = {"g": str(fps), "bf": "0", "tune": "zerolatency"}
        audio = out.add_stream("pcm_mulaw", rate=8000, layout="mono")
        started = time.monotonic()
        for i in range(int(seconds * fps)):
            image = np.zeros((height, width, 3), np.uint8)
            image[:, (i * 4) % (width - 20) :][:, :20] = 255
            frame = av.VideoFrame.from_ndarray(image, format="rgb24")
            frame.pts, frame.time_base = i, Fraction(1, fps)
            out.mux(video.encode(frame))
            samples = 8000 // fps
            tone = (np.sin(np.arange(i * samples, (i + 1) * samples) * 0.2) * 8000).astype(np.int16)
            sound = av.AudioFrame.from_ndarray(tone[None, :], format="s16", layout="mono")
            sound.sample_rate, sound.pts, sound.time_base = 8000, i * samples, Fraction(1, 8000)
            out.mux(audio.encode(sound))
            time.sleep(max(0.0, started + (i + 1) / fps - time.monotonic()))
        out.mux(video.encode())
        out.mux(audio.encode())


@pytest.mark.skipif(not (RTSP and PLAYBACK), reason="no MediaMTX for integration tests")
def test_recording_reconnect_gap_and_clip(tmp_path: Path) -> None:
    started = datetime.now(UTC)
    publish(20.0)
    time.sleep(8.0)  # the camera is away
    publish(15.0)
    ended = datetime.now(UTC)
    time.sleep(3.0)  # let the last part reach the disk

    spans = [s for s in list_spans(PLAYBACK, PATH) if s.end > started]
    assert len(spans) == 2
    hole_s = (spans[1].start - spans[0].end).total_seconds()
    assert 5.0 <= hole_s <= 12.0

    result = export_clip(
        PLAYBACK, PATH, started, ended, tmp_path / "clip.mp4", max_bytes=50_000_000
    )
    info = probe(result.path)
    assert info.audio_codec == "aac" and info.video_codec == "h264"
    assert len(result.gaps) == 1
    assert result.gaps[0].start == spans[0].end
    assert result.actual_end <= ended
    assert result.actual_start <= max(started, spans[0].start) + timedelta(seconds=1)
