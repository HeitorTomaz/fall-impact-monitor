"""A local HTTP server standing in for the MediaMTX playback server."""

import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import pytest

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
