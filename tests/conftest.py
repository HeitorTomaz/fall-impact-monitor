"""Shared fixtures. Media is synthesized at test time; no recordings are used."""

import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import av
import numpy as np
import pytest

from tests.database import database  # noqa: F401  (fixture for every test package)

SYNTH_FPS = 25
SYNTH_FRAMES = 75
SYNTH_SIZE = (640, 360)


def write_synthetic_video(
    path: Path, frames: int = SYNTH_FRAMES, fps: int = SYNTH_FPS, b_frames: int = 0
) -> Path:
    """Encode a moving white bar as H.264 with one keyframe per second and no B-frames."""
    width, height = SYNTH_SIZE
    with av.open(str(path), "w") as out:
        stream = out.add_stream("libx264", rate=fps)
        stream.width, stream.height, stream.pix_fmt = width, height, "yuv420p"
        stream.options = {
            "g": str(fps),
            "keyint_min": str(fps),
            "sc_threshold": "0",
            "bf": str(b_frames),
        }
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


# A local HTTP server standing in for the MediaMTX playback server.
Response = tuple[int, bytes]
Route = Callable[[dict[str, str]], Response]


@dataclass
class FakePlayback:
    url: str = ""
    routes: dict[str, Route] = field(default_factory=dict)
    requests: list[tuple[str, dict[str, str]]] = field(default_factory=list)


@pytest.fixture
def playback() -> Iterator[FakePlayback]:
    fake = FakePlayback()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            query = {k: v[0] for k, v in parse_qs(parsed.query).items()}
            fake.requests.append((parsed.path, query))
            route = fake.routes.get(parsed.path)
            status, body = route(query) if route else (404, b"not found")
            self.send_response(status)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    fake.url = f"http://127.0.0.1:{server.server_address[1]}"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield fake
    server.shutdown()
    server.server_close()
