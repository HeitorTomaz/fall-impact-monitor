"""`fim zones --edit`: draw and edit zone polygons with the mouse, on the camera snapshot.

A small local web page served by the standard library: it loads ``snapshot.png`` and
``zones.json`` from the zone file's folder, lets the installer move, add and delete
points and zones, rename them and accept suggestions (``suggested_safe`` → ``safe``), and
saves through the same validation the service uses; ``zones.png`` is redrawn on every
save. Nothing is loaded from outside and nothing leaves the machine except to the browser
that opens the page.

Access: it listens on 127.0.0.1 unless told otherwise; every request must carry the random
token printed with the URL, and the ``Host`` header must be a local name, so that another
web page open in the same browser cannot read the snapshot of the home or change the zones
(cross-site requests, DNS rebinding).
"""

import json
import os
import secrets
import tempfile
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from fall_impact_monitor.analysis.zones import SUGGESTED_KIND, load_zones

_MAX_BODY_BYTES = 1_000_000
_LOCAL_HOSTS = frozenset({"127.0.0.1", "localhost", "[::1]", "::1"})
_PAGE = Path(__file__).with_name("zone_editor.html")  # shipped in the package
_SAVE_LOCK = threading.Lock()  # one save at a time: file and overview stay consistent


def _host_name(header: str) -> str:
    """``Host`` header without its port (``[::1]:8090`` -> ``[::1]``)."""
    if header.startswith("["):
        return header.split("]")[0] + "]"
    return header.rsplit(":", 1)[0] if header.count(":") == 1 else header


def make_handler(
    zone_file: Path, token: str, hosts: frozenset[str] = _LOCAL_HOSTS
) -> type[BaseHTTPRequestHandler]:
    folder = zone_file.parent

    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Frame-Options", "DENY")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status: int, value: object) -> None:
            self._send(status, json.dumps(value).encode(), "application/json")

        def _allowed(self) -> str | None:
            """The request path when the host and the token are right, else None (403)."""
            if _host_name(self.headers.get("Host", "")) not in hosts:
                self._json(HTTPStatus.FORBIDDEN, {"error": "not a local address"})
                return None
            parts = urlsplit(self.path)
            given = parse_qs(parts.query).get("token", [""])[0]
            if not secrets.compare_digest(given, token):
                self._json(HTTPStatus.FORBIDDEN, {"error": "missing or wrong token"})
                return None
            return parts.path

        def do_GET(self) -> None:
            path = self._allowed()
            if path is None:
                return
            if path in ("/", "/index.html"):
                self._send(HTTPStatus.OK, _PAGE.read_bytes(), "text/html; charset=utf-8")
            elif path == "/snapshot.png" and (folder / "snapshot.png").exists():
                self._send(HTTPStatus.OK, (folder / "snapshot.png").read_bytes(), "image/png")
            elif path == "/zones.json":
                self._send(HTTPStatus.OK, zone_file.read_bytes(), "application/json")
            else:
                self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})

        def do_POST(self) -> None:
            path = self._allowed()
            if path is None:
                return
            if path != "/zones.json":
                self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
                return
            if not self.headers.get("Content-Type", "").startswith("application/json"):
                self._json(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, {"error": "JSON only"})
                return
            try:
                length = int(self.headers.get("Content-Length", ""))
            except ValueError:
                length = -1
            if not 0 <= length <= _MAX_BODY_BYTES:
                self._json(HTTPStatus.BAD_REQUEST, {"error": "bad or too large body"})
                return
            try:
                reply = save(zone_file, json.loads(self.rfile.read(length)))
            except (ValueError, KeyError, TypeError, AttributeError) as error:
                self._json(HTTPStatus.BAD_REQUEST, {"error": str(error) or type(error).__name__})
            except Exception as error:  # e.g. a folder it cannot write: say so, keep serving
                self._json(
                    HTTPStatus.INTERNAL_SERVER_ERROR, {"error": f"{type(error).__name__}: {error}"}
                )
            else:
                self._json(HTTPStatus.OK, reply)

        def log_message(self, format: str, *args: object) -> None:
            pass  # keep the terminal for the URL

    return Handler


def save(zone_file: Path, data: object) -> dict[str, object]:
    """Validate ``data`` like the service does, then replace the file and redraw zones.png.
    Raises and leaves the file untouched when the zones are invalid."""
    from fall_impact_monitor.cli.zones import write_overview

    if not isinstance(data, dict) or not isinstance(data.get("zones"), list):
        raise ValueError('expected {"zones": [...]}')
    try:
        existing = json.loads(zone_file.read_text(encoding="utf-8"))
        frame_size, rotate_deg = existing.get("frame_size"), existing.get("rotate_deg", 0)
    except (OSError, ValueError, AttributeError):
        frame_size, rotate_deg = None, 0  # a damaged file can still be replaced by valid zones
    content = {
        "frame_size": data.get("frame_size", frame_size),
        "rotate_deg": rotate_deg,  # the snapshot's rotation; the page does not change it
        "zones": data["zones"],
    }
    with _SAVE_LOCK:
        handle, name = tempfile.mkstemp(prefix=".zones-", suffix=".tmp", dir=zone_file.parent)
        temporary = Path(name)
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as out:
                out.write(json.dumps(content, indent=2) + "\n")
            zones = load_zones(temporary, allow_suggested=True)
            temporary.chmod(0o644)  # the service reads it as another user
            os.replace(temporary, zone_file)
        finally:
            temporary.unlink(missing_ok=True)
        drawn = write_overview(zone_file, zones)
    return {
        "ok": True,
        "overview": drawn,
        "suggested": [z.name for z in zones if z.kind == SUGGESTED_KIND],
    }


def serve(zone_file: Path, host: str, port: int) -> int:
    token = secrets.token_urlsafe(16)
    hosts = _LOCAL_HOSTS if host in ("0.0.0.0", "::") else _LOCAL_HOSTS | {host}
    server = ThreadingHTTPServer((host, port), make_handler(zone_file, token, hosts))
    shown = "127.0.0.1" if host in ("0.0.0.0", "::") else host
    print(f"zones: editing {zone_file}")
    print(f"zones: open http://{shown}:{server.server_address[1]}/?token={token} (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0
