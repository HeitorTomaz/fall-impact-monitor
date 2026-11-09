"""MediaMTX configuration generated from the camera variables (single source of truth).

One upstream connection per camera stream: the main stream is relayed and recorded as
fMP4 segments under ``/recordings/<path>/``; the optional substream is relayed for
analysis only. The playback server serves recorded time ranges to clip export. Nothing
is published outside the Docker network.

The file is JSON, which MediaMTX reads as YAML. It contains the camera URLs (MediaMTX
needs them), so it is written readable by its owner only and never printed.
"""

import json
import os
from pathlib import Path
from typing import Any

from fall_impact_monitor.config import CameraSettings

# A camera URL with this value makes the path accept a publisher instead of pulling a
# camera: used by the simulated camera during development and by integration tests.
PUBLISHER = "publisher"

_BASE: dict[str, Any] = {
    "logLevel": "info",
    "api": False,
    "metrics": False,
    "pprof": False,
    "playback": True,
    "playbackAddress": ":9996",
    "rtsp": True,
    "rtspTransports": ["tcp"],
    "rtspAddress": ":8554",
    "rtmp": False,
    "hls": False,
    "webrtc": False,
    "srt": False,
    "moq": False,
}

_PATH_DEFAULTS: dict[str, Any] = {
    # Pull camera streams over TCP and keep camera timestamps (no absolute time).
    "rtspTransport": "tcp",
    "useAbsoluteTimestamp": False,
    "record": False,
    "recordPath": "/recordings/%path/%Y-%m-%d_%H-%M-%S-%f_%z",
    "recordFormat": "fmp4",
    "recordPartDuration": "1s",
    "recordSegmentDuration": "1m",
}


def substream_path(camera_id: str) -> str:
    """MediaMTX path name of a camera's substream."""
    return f"{camera_id}sub"


def build_mediamtx_config(cameras: list[CameraSettings], *, max_age_h: float) -> dict[str, Any]:
    """Main stream path = camera id (recorded); substream path only when one is set."""
    paths: dict[str, dict[str, Any]] = {}
    for camera in cameras:
        paths[camera.id] = {"source": camera.main_url, "record": True}
        if camera.sub_url:
            paths[substream_path(camera.id)] = {"source": camera.sub_url, "record": False}
    return {
        **_BASE,
        "pathDefaults": {**_PATH_DEFAULTS, "recordDeleteAfter": f"{max_age_h:g}h"},
        "paths": paths,
    }


def write_mediamtx_config(config: dict[str, Any], out: Path) -> None:
    """Write atomically, readable and writable by the owner only."""
    out.parent.mkdir(parents=True, exist_ok=True)
    partial = out.with_name(out.name + ".part")
    fd = os.open(partial, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
        handle.write("\n")
    partial.replace(out)
