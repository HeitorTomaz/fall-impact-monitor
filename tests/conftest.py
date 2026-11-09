"""Shared fixtures. Media is synthesized at test time; no recordings are used."""

from pathlib import Path

import av
import numpy as np
import pytest

SYNTH_FPS = 25
SYNTH_FRAMES = 75
SYNTH_SIZE = (640, 360)


def write_synthetic_video(path: Path, frames: int = SYNTH_FRAMES, fps: int = SYNTH_FPS) -> Path:
    """Encode a moving white bar as H.264 with one keyframe per second and no B-frames."""
    width, height = SYNTH_SIZE
    with av.open(str(path), "w") as out:
        stream = out.add_stream("libx264", rate=fps)
        stream.width, stream.height, stream.pix_fmt = width, height, "yuv420p"
        stream.options = {"g": str(fps), "keyint_min": str(fps), "sc_threshold": "0", "bf": "0"}
        for i in range(frames):
            image = np.zeros((height, width, 3), np.uint8)
            x = (i * 8) % (width - 40)
            image[:, x : x + 40] = 255
            for packet in stream.encode(av.VideoFrame.from_ndarray(image, format="rgb24")):
                out.mux(packet)
        for packet in stream.encode():
            out.mux(packet)
    return path


@pytest.fixture(scope="session")
def synthetic_video(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return write_synthetic_video(tmp_path_factory.mktemp("media") / "synthetic.mp4")


@pytest.fixture(scope="session")
def synthetic_person_frame() -> "np.ndarray":
    """A plain BGR frame; only checks that real models run end to end."""
    return np.full((360, 640, 3), 128, np.uint8)
