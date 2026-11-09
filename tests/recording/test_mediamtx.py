import threading
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlsplit

import pytest

from fall_impact_monitor.recording import fetch_clip, playback_get_url

PAYLOAD = b"\x00\x00\x00\x18ftypmp42fake-mp4-bytes"


def test_get_url_converts_start_to_utc_and_encodes_query() -> None:
    start = datetime(2026, 11, 9, 1, 2, 3, tzinfo=timezone(timedelta(hours=-3)))
    url = playback_get_url("http://mediamtx:9996", "cam1", start, 30)
    parts = urlsplit(url)
    assert f"{parts.scheme}://{parts.netloc}{parts.path}" == "http://mediamtx:9996/get"
    assert parse_qs(parts.query) == {
        "path": ["cam1"],
        "start": ["2026-11-09T04:02:03Z"],
        "duration": ["30"],
        "format": ["mp4"],
    }


def test_naive_start_is_rejected() -> None:
    with pytest.raises(ValueError, match="timezone"):
        playback_get_url("http://mediamtx:9996", "cam1", datetime(2026, 1, 1), 10)


def test_non_positive_duration_is_rejected() -> None:
    with pytest.raises(ValueError, match="duration"):
        playback_get_url("http://mediamtx:9996", "cam1", datetime(2026, 1, 1, tzinfo=UTC), 0)


@pytest.fixture()
def server() -> Iterator[str]:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path.startswith("/get?"):
                self.send_response(200)
                self.send_header("Content-Type", "video/mp4")
                self.end_headers()
                self.wfile.write(PAYLOAD)
            else:
                self.send_error(404)

        def log_message(self, *args: object) -> None:
            pass

    httpd = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_port}"
    httpd.shutdown()


def test_fetch_clip_writes_the_response(server: str, tmp_path: Path) -> None:
    url = playback_get_url(server, "cam1", datetime(2026, 1, 1, tzinfo=UTC), 10)
    dest = fetch_clip(url, tmp_path / "clips" / "clip.mp4")
    assert dest.read_bytes() == PAYLOAD


def test_fetch_clip_raises_on_http_error_and_leaves_no_file(server: str, tmp_path: Path) -> None:
    with pytest.raises(HTTPError):
        fetch_clip(f"{server}/missing", tmp_path / "clip.mp4")
    assert not (tmp_path / "clip.mp4").exists()
