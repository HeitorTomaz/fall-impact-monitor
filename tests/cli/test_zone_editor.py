import json
import threading
import urllib.error
import urllib.request
from collections.abc import Iterator
from http.server import ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np
import pytest

from fall_impact_monitor.analysis.zones import load_zones
from fall_impact_monitor.cli.zone_editor import make_handler

BED = {"name": "bed_1", "kind": "suggested_safe", "polygon": [[0.1, 0.1], [0.5, 0.1], [0.5, 0.5]]}
TOKEN = "secret-token"


@pytest.fixture
def editor(tmp_path: Path) -> Iterator[tuple[str, Path]]:
    cv2.imwrite(str(tmp_path / "snapshot.png"), np.zeros((36, 64, 3), np.uint8))
    zone_file = tmp_path / "zones.json"
    zone_file.write_text(json.dumps({"frame_size": [64, 36], "zones": [BED]}))
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(zone_file, TOKEN))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}", zone_file
    finally:
        server.shutdown()


def request(
    url: str,
    body: bytes | None = None,
    headers: dict[str, str] | None = None,
) -> tuple[int, bytes, str]:
    method = "GET" if body is None else "POST"
    req = urllib.request.Request(url, body, headers or {}, method=method)
    try:
        with urllib.request.urlopen(req) as response:
            return response.status, response.read(), response.headers["Content-Type"]
    except urllib.error.HTTPError as error:
        return error.code, error.read(), error.headers["Content-Type"]


def save(url: str, zones: object, content_type: str = "application/json") -> tuple[int, dict]:  # type: ignore[type-arg]
    body = json.dumps({"zones": zones}).encode()
    status, reply, _ = request(
        f"{url}/zones.json?token={TOKEN}", body, {"Content-Type": content_type}
    )
    return status, json.loads(reply or b"{}")


def test_the_page_the_snapshot_and_the_zones_are_served_with_the_token(
    editor: tuple[str, Path],
) -> None:
    url, _ = editor
    status, page, kind = request(f"{url}/?token={TOKEN}")
    assert status == 200 and kind.startswith("text/html") and b"<canvas" in page
    assert b"http://" not in page and b"https://" not in page  # nothing loaded from outside
    status, image, kind = request(f"{url}/snapshot.png?token={TOKEN}")
    assert status == 200 and kind == "image/png" and image.startswith(b"\x89PNG")
    status, zones, _ = request(f"{url}/zones.json?token={TOKEN}")
    assert json.loads(zones)["zones"] == [BED]


def test_requests_without_the_token_are_refused(editor: tuple[str, Path]) -> None:
    url, zone_file = editor
    before = zone_file.read_text()
    for path in ("/", "/snapshot.png", "/zones.json", "/zones.json?token=wrong"):
        assert request(url + path)[0] == 403
    body = json.dumps({"zones": []}).encode()
    status, _, _ = request(f"{url}/zones.json", body, {"Content-Type": "application/json"})
    assert status == 403 and zone_file.read_text() == before


def test_a_foreign_host_name_is_refused_even_with_the_token(editor: tuple[str, Path]) -> None:
    # DNS rebinding: a web page whose name now points at 127.0.0.1.
    url, _ = editor
    status, _, _ = request(f"{url}/snapshot.png?token={TOKEN}", headers={"Host": "evil.example"})
    assert status == 403


def test_saving_valid_zones_writes_the_file_and_redraws_the_overview(
    editor: tuple[str, Path],
) -> None:
    url, zone_file = editor
    accepted = {**BED, "kind": "safe", "polygon": [[0.1, 0.1], [0.4, 0.1], [0.4, 0.4]]}
    status, reply = save(url, [accepted])
    assert status == 200 and reply["suggested"] == [] and reply["overview"] is True
    assert [z.kind for z in load_zones(zone_file)] == ["safe"]
    assert (zone_file.parent / "zones.png").stat().st_size > 0
    assert json.loads(zone_file.read_text())["frame_size"] == [64, 36]
    assert not list(zone_file.parent.glob("*.tmp"))


def test_suggestions_can_be_saved_but_are_reported(editor: tuple[str, Path]) -> None:
    url, _ = editor
    status, reply = save(url, [BED])
    assert status == 200 and reply["suggested"] == ["bed_1"]


def test_invalid_zones_are_refused_and_the_file_is_kept(editor: tuple[str, Path]) -> None:
    url, zone_file = editor
    before = zone_file.read_text()
    status, reply = save(url, [{"name": "x", "kind": "safe", "polygon": [[0.1, 0.1], [0.2, 0.2]]}])
    assert status == 400 and "3 points" in reply["error"]
    status, reply = save(url, ["not a zone"])
    assert status == 400 and reply["error"]
    assert zone_file.read_text() == before
    assert not list(zone_file.parent.glob("*.tmp"))


def test_only_json_is_accepted(editor: tuple[str, Path]) -> None:
    url, zone_file = editor
    before = zone_file.read_text()
    status, _ = save(url, [], "application/x-www-form-urlencoded")
    assert status == 415 and zone_file.read_text() == before


def test_a_bad_content_length_is_refused(editor: tuple[str, Path]) -> None:
    import http.client
    from urllib.parse import urlparse

    url, _ = editor
    address = urlparse(url)
    connection = http.client.HTTPConnection(address.hostname or "", address.port)
    connection.putrequest("POST", f"/zones.json?token={TOKEN}")
    connection.putheader("Content-Type", "application/json")
    connection.putheader("Content-Length", "-1")
    connection.endheaders()
    assert connection.getresponse().status == 400


def test_the_editor_needs_an_existing_zone_file(tmp_path: Path) -> None:
    from fall_impact_monitor.cli.main import main

    assert main(["zones", "--edit", str(tmp_path / "missing.json")]) == 1
