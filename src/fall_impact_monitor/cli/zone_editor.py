"""`fim zones --edit`: draw and edit zone polygons with the mouse, on the camera snapshot.

A small local web page served by the standard library: it loads ``snapshot.png`` and
``zones.json`` from the zone file's folder, lets the installer move, add and delete
points and zones, rename them and accept suggestions (``suggested_safe`` → ``safe``), and
saves through the same validation the service uses; ``zones.png`` is redrawn on every
save. Nothing is loaded from outside and nothing leaves the machine except to the browser
that opens the page. It listens on 127.0.0.1 unless told otherwise.
"""

import json
import os
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from fall_impact_monitor.analysis.zones import SUGGESTED_KIND, load_zones

_MAX_BODY_BYTES = 1_000_000

_PAGE = Path(__file__).with_name("zone_editor.html")  # shipped in the package


def make_handler(zone_file: Path) -> type[BaseHTTPRequestHandler]:
    folder = zone_file.parent

    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status: int, value: object) -> None:
            self._send(status, json.dumps(value).encode(), "application/json")

        def do_GET(self) -> None:
            if self.path in ("/", "/index.html"):
                self._send(HTTPStatus.OK, _PAGE.read_bytes(), "text/html; charset=utf-8")
            elif self.path == "/snapshot.png" and (folder / "snapshot.png").exists():
                self._send(HTTPStatus.OK, (folder / "snapshot.png").read_bytes(), "image/png")
            elif self.path == "/zones.json":
                self._send(HTTPStatus.OK, zone_file.read_bytes(), "application/json")
            else:
                self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})

        def do_POST(self) -> None:
            if self.path != "/zones.json":
                self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
                return
            if not self.headers.get("Content-Type", "").startswith("application/json"):
                self._json(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, {"error": "JSON only"})
                return
            length = int(self.headers.get("Content-Length", 0))
            if length > _MAX_BODY_BYTES:
                self._json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": "too large"})
                return
            try:
                self._json(HTTPStatus.OK, save(zone_file, json.loads(self.rfile.read(length))))
            except (ValueError, KeyError, TypeError) as error:
                self._json(HTTPStatus.BAD_REQUEST, {"error": str(error)})

        def log_message(self, format: str, *args: object) -> None:
            pass  # the page polls nothing; keep the terminal for the URL

    return Handler


def save(zone_file: Path, data: object) -> dict[str, object]:
    """Validate ``data`` like the service does, then replace the file and redraw zones.png.
    Raises ``ValueError`` and leaves the file untouched when the zones are invalid."""
    from fall_impact_monitor.cli.zones import write_overview

    if not isinstance(data, dict) or not isinstance(data.get("zones"), list):
        raise ValueError('expected {"zones": [...]}')
    current = json.loads(zone_file.read_text(encoding="utf-8"))
    content = {
        "frame_size": data.get("frame_size", current.get("frame_size")),
        "zones": data["zones"],
    }
    temporary = zone_file.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(content, indent=2) + "\n", encoding="utf-8")
    try:
        zones = load_zones(temporary, allow_suggested=True)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    os.replace(temporary, zone_file)
    write_overview(zone_file, zones)
    return {"ok": True, "suggested": [z.name for z in zones if z.kind == SUGGESTED_KIND]}


def serve(zone_file: Path, host: str, port: int) -> int:
    server = ThreadingHTTPServer((host, port), make_handler(zone_file))
    print(
        f"zones: editing {zone_file} at http://{host}:{server.server_address[1]}/ (Ctrl+C to stop)"
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0
