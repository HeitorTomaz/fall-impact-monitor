"""Synthetic camera-like media for clip tests: H.264 video, optional G711 audio."""

from fractions import Fraction
from pathlib import Path

import av
import numpy as np

WIDTH, HEIGHT, FPS = 320, 240, 15


def write_camera_clip(
    path: Path, seconds: float, *, audio: bool = True, noise: bool = False
) -> Path:
    """``noise`` encodes random frames losslessly, which makes a large file."""
    rng = np.random.default_rng(0)
    with av.open(str(path), "w") as out:
        video = out.add_stream("libx264", rate=FPS)
        video.width, video.height, video.pix_fmt = WIDTH, HEIGHT, "yuv420p"
        video.options = {"g": str(FPS), "bf": "0", **({"qp": "0"} if noise else {})}
        sound = out.add_stream("pcm_mulaw", rate=8000, layout="mono") if audio else None
        for i in range(int(seconds * FPS)):
            if noise:
                image = rng.integers(0, 255, (HEIGHT, WIDTH, 3), np.uint8)
            else:
                image = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
                image[:, (i * 4) % (WIDTH - 20) :][:, :20] = 255
            frame = av.VideoFrame.from_ndarray(image, format="rgb24")
            frame.pts, frame.time_base = i, Fraction(1, FPS)
            out.mux(video.encode(frame))
        out.mux(video.encode())
        if sound is not None:
            samples = int(seconds * 8000)
            for start in range(0, samples, 800):
                n = min(800, samples - start)
                tone = (np.sin(np.arange(start, start + n) * 0.2) * 8000).astype(np.int16)
                frame = av.AudioFrame.from_ndarray(tone[None, :], format="s16", layout="mono")
                frame.sample_rate, frame.pts = 8000, start
                frame.time_base = Fraction(1, 8000)
                out.mux(sound.encode(frame))
            out.mux(sound.encode())
    return path
