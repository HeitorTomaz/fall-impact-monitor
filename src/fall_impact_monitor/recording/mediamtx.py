"""Client helpers for the MediaMTX playback server (clip export)."""

import shutil
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlencode


def playback_get_url(
    api_url: str, path: str, start: datetime, duration_s: float, fmt: str = "mp4"
) -> str:
    """URL of the playback ``/get`` endpoint for ``duration_s`` seconds from ``start``.

    MediaMTX returns a single file that begins at the GOP containing ``start``.
    """
    if start.tzinfo is None:
        raise ValueError("start must include a timezone")
    if duration_s <= 0:
        raise ValueError("duration must be positive")
    start_utc = start.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ").replace(".000000Z", "Z")
    duration = f"{duration_s:g}"
    query = urlencode({"path": path, "start": start_utc, "duration": duration, "format": fmt})
    return f"{api_url.rstrip('/')}/get?{query}"


def fetch_clip(url: str, dest: Path, timeout_s: float = 60.0) -> Path:
    """Download ``url`` to ``dest`` atomically; raise on HTTP errors without leaving a file."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_name(dest.name + ".part")
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as response, partial.open("wb") as out:
            shutil.copyfileobj(response, out)
        partial.replace(dest)
    finally:
        partial.unlink(missing_ok=True)
    return dest
