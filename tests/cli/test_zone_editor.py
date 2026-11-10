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


@pytest.fixture
def editor(tmp_path: Path) -> Iterator[tuple[str, Path]]:
    cv2.imwrite(str(tmp_path / "snapshot.png"), np.zeros((36, 64, 3), np.uint8))
    zone_file = tmp_path / "zones.json"
    zone_file.write_text(json.dumps({"frame_size": [64, 36], "zones": [BED]}))
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(zone_file))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}", zone_file
    finally:
        server.shutdown()


def get(url: str) -> tuple[int, bytes, str]:
    with urllib.request.urlopen(url) as response:
        return response.status, response.read(), response.headers["Content-Type"]


def post(url: str, body: bytes, content_type: str = "application/json") -> tuple[int, dict]:  # type: ignore[type-arg]
    request = urllib.request.Request(url, body, {"Content-Type": content_type}, method="POST")
    try:
        with urllib.request.urlopen(request) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read() or b"{}")


def test_the_page_the_snapshot_and_the_zones_are_served(editor: tuple[str, Path]) -> None:
    url, _ = editor
    status, page, kind = get(url + "/")
    assert status == 200 and kind.startswith("text/html") and b"<canvas" in page
    assert b"http://" not in page and b"https://" not in page  # nothing loaded from outside
    status, image, kind = get(url + "/snapshot.png")
    assert status == 200 and kind == "image/png" and image.startswith(b"\x89PNG")
    status, zones, _ = get(url + "/zones.json")
    assert json.loads(zones)["zones"] == [BED]


def test_saving_valid_zones_writes_the_file_and_redraws_the_overview(
    editor: tuple[str, Path],
) -> None:
    url, zone_file = editor
    accepted = {**BED, "kind": "safe", "polygon": [[0.1, 0.1], [0.4, 0.1], [0.4, 0.4]]}
    status, reply = post(
        url + "/zones.json", json.dumps({"frame_size": [64, 36], "zones": [accepted]}).encode()
    )
    assert status == 200 and reply["suggested"] == []
    assert [z.kind for z in load_zones(zone_file)] == ["safe"]
    assert (zone_file.parent / "zones.png").stat().st_size > 0


def test_suggestions_can_be_saved_but_are_reported(editor: tuple[str, Path]) -> None:
    url, _ = editor
    status, reply = post(url + "/zones.json", json.dumps({"zones": [BED]}).encode())
    assert status == 200 and reply["suggested"] == ["bed_1"]


def test_invalid_zones_are_refused_and_the_file_is_kept(editor: tuple[str, Path]) -> None:
    url, zone_file = editor
    before = zone_file.read_text()
    bad = {"name": "x", "kind": "safe", "polygon": [[0.1, 0.1], [0.2, 0.2]]}
    status, reply = post(url + "/zones.json", json.dumps({"zones": [bad]}).encode())
    assert status == 400 and "3 points" in reply["error"]
    assert zone_file.read_text() == before


def test_only_json_is_accepted(editor: tuple[str, Path]) -> None:
    url, zone_file = editor
    before = zone_file.read_text()
    status, _ = post(url + "/zones.json", b"zones=1", "application/x-www-form-urlencoded")
    assert status == 415 and zone_file.read_text() == before


def test_the_editor_needs_an_existing_zone_file(tmp_path: Path) -> None:
    from fall_impact_monitor.cli.main import main

    assert main(["zones", "--edit", str(tmp_path / "missing.json")]) == 1
