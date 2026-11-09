"""Client helpers for the MediaMTX playback server: recorded spans and clip download."""

import json
import re
import shutil
import time
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode

_FRACTION = re.compile(r"(\.\d{6})\d+")


@dataclass(frozen=True)
class Span:
    """A recorded (or missing) time range, UTC-aware, ``start <= end``."""

    start: datetime
    end: datetime


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("datetimes must include a timezone")
    return value.astimezone(UTC)


def _rfc3339(value: datetime) -> str:
    return _utc(value).strftime("%Y-%m-%dT%H:%M:%S.%fZ").replace(".000000Z", "Z")


def _parse_rfc3339(text: str) -> datetime:
    # MediaMTX may print nanoseconds; datetime keeps microseconds.
    return _utc(datetime.fromisoformat(_FRACTION.sub(r"\1", text)))


def list_spans(
    api_url: str,
    path: str,
    start: datetime | None = None,
    end: datetime | None = None,
    timeout_s: float = 10.0,
) -> list[Span]:
    """Recorded spans of ``path`` from the playback ``/list`` endpoint, oldest first.

    MediaMTX merges contiguous segments into one span, so the space between two spans
    is a gap in the recording. Nothing recorded (HTTP 404) is an empty list.
    """
    query = {"path": path}
    if start is not None:
        query["start"] = _rfc3339(start)
    if end is not None:
        query["end"] = _rfc3339(end)
    url = f"{api_url.rstrip('/')}/list?{urlencode(query)}"
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as response:
            items = json.load(response)
    except HTTPError as error:
        if error.code == 404:
            return []
        raise
    spans = []
    for item in items:
        span_start = _parse_rfc3339(str(item["start"]))
        spans.append(Span(span_start, span_start + timedelta(seconds=float(item["duration"]))))
    return sorted(spans, key=lambda s: s.start)


def gaps(spans: list[Span], start: datetime, end: datetime, min_gap_s: float = 2.0) -> list[Span]:
    """Parts of ``[start, end]`` not covered by ``spans`` that last at least ``min_gap_s``."""
    start, end = _utc(start), _utc(end)
    missing: list[Span] = []
    cursor = start
    for span in sorted(spans, key=lambda s: s.start):
        if span.end <= cursor:
            continue
        if span.start >= end:
            break
        if span.start > cursor:
            missing.append(Span(cursor, span.start))
        cursor = max(cursor, span.end)
    if cursor < end:
        missing.append(Span(cursor, end))
    return [g for g in missing if (g.end - g.start).total_seconds() >= min_gap_s]


def wait_for_coverage(
    api_url: str,
    path: str,
    until: datetime,
    *,
    timeout_s: float,
    poll_s: float = 2.0,
    clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    sleep: Callable[[float], None] | None = None,
) -> bool:
    """Block until the recording of ``path`` reaches ``until``; False at the timeout.

    Used before exporting an event clip, so that the post-event window is on disk. An
    unreachable playback server counts as "not yet" (MediaMTX may be restarting).
    """
    sleep = sleep or time.sleep
    until = _utc(until)
    deadline = clock() + timedelta(seconds=timeout_s)
    while True:
        try:
            if any(span.end >= until for span in list_spans(api_url, path)):
                return True
        except (URLError, OSError):
            pass
        remaining = (deadline - clock()).total_seconds()
        if remaining <= 0:
            return False
        sleep(min(poll_s, remaining))


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
    start_utc = _rfc3339(start)
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
