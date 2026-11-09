"""Clip export from MediaMTX recordings, with the requested and the actual windows.

The playback server returns the recorded media of a time range, starting at the
keyframe at or before the requested start, and skipping gaps in the recording. The
actual window is therefore derived, never assumed: it ends where the request or the
recording ends, and it starts earlier than the first recorded instant by however much
the file is longer than the recorded part of the window (the keyframe snap). Gaps are
listed explicitly.

Each clip has a JSON sidecar with both windows, so the information survives without
the database.
"""

import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from fall_impact_monitor.recording.clip_media import fit_clip, probe
from fall_impact_monitor.recording.mediamtx import (
    Span,
    fetch_clip,
    gaps,
    list_spans,
    playback_get_url,
    wait_for_coverage,
)


class ClipUnavailable(Exception):
    """Nothing was recorded in the requested window."""


@dataclass(frozen=True)
class ClipResult:
    path: Path
    camera_id: str
    requested_start: datetime
    requested_end: datetime
    actual_start: datetime
    actual_end: datetime
    size_bytes: int
    transcoded: bool
    has_audio: bool
    gaps: list[Span]

    def to_json(self) -> dict[str, Any]:
        return {
            "camera_id": self.camera_id,
            "file": self.path.name,
            "requested_start": self.requested_start.isoformat(),
            "requested_end": self.requested_end.isoformat(),
            "actual_start": self.actual_start.isoformat(),
            "actual_end": self.actual_end.isoformat(),
            "size_bytes": self.size_bytes,
            "transcoded": self.transcoded,
            "has_audio": self.has_audio,
            "gaps": [{"start": g.start.isoformat(), "end": g.end.isoformat()} for g in self.gaps],
        }


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("clip times must include a timezone")
    return value.astimezone(UTC)


def export_clip(
    api_url: str, camera_id: str, start: datetime, end: datetime, dest: Path, *, max_bytes: int
) -> ClipResult:
    """Export ``[start, end]`` of camera ``camera_id`` (its MediaMTX path) to ``dest``
    plus ``dest.with_suffix('.json')``. Raises ``ClipUnavailable`` when nothing in the
    window was recorded; on any error, neither file is left behind."""
    start, end = _utc(start), _utc(end)
    if end <= start:
        raise ValueError("clip end must be after its start")
    covering = [s for s in list_spans(api_url, camera_id) if s.end > start and s.start < end]
    if not covering:
        raise ClipUnavailable(f"nothing recorded for {camera_id} in the requested window")
    download = dest.with_name(dest.stem + ".download.mp4")
    sidecar = dest.with_suffix(".json")
    try:
        url = playback_get_url(api_url, camera_id, start, (end - start).total_seconds())
        fetch_clip(url, download)
        transcoded = fit_clip(download, dest, max_bytes)
        info = probe(dest)
        recorded_s = sum((min(s.end, end) - max(s.start, start)).total_seconds() for s in covering)
        snap_s = max(0.0, info.duration_s - recorded_s)
        actual_start = max(start, covering[0].start) - timedelta(seconds=snap_s)
        actual_end = min(end, covering[-1].end)
        result = ClipResult(
            path=dest,
            camera_id=camera_id,
            requested_start=start,
            requested_end=end,
            actual_start=actual_start,
            actual_end=actual_end,
            size_bytes=info.size_bytes,
            transcoded=transcoded,
            has_audio=info.audio_codec is not None,
            gaps=gaps(covering, start, end),
        )
        partial = sidecar.with_name(sidecar.name + ".part")
        partial.write_text(json.dumps(result.to_json(), indent=2) + "\n", encoding="utf-8")
        partial.replace(sidecar)
    except BaseException:
        dest.unlink(missing_ok=True)
        sidecar.unlink(missing_ok=True)
        sidecar.with_name(sidecar.name + ".part").unlink(missing_ok=True)
        raise
    finally:
        download.unlink(missing_ok=True)
    return result


def event_clip_path(events_dir: Path, camera_id: str, start: datetime, event_id: str) -> Path:
    """``<events_dir>/<camera>/<YYYY>/<MM>/<DD>/<HHMMSS>_<event_id>.mp4`` in UTC."""
    t = _utc(start)
    return events_dir / camera_id / f"{t:%Y}" / f"{t:%m}" / f"{t:%d}" / f"{t:%H%M%S}_{event_id}.mp4"


class EventClipExporter:
    """Exports ``[start - pre_s, end + post_s]`` of an event once it is recorded.

    ``export`` blocks until the recording reaches the end of the window (or
    ``wait_timeout_s`` after it, then exports what exists; the sidecar lists the gap).
    The monitoring service calls it from a worker, never from the alert path.
    """

    def __init__(
        self,
        api_url: str,
        events_dir: Path,
        *,
        pre_s: float,
        post_s: float,
        max_bytes: int,
        wait_timeout_s: float = 120.0,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        self._api_url = api_url
        self._events_dir = events_dir
        self._pre = timedelta(seconds=pre_s)
        self._post = timedelta(seconds=post_s)
        self._max_bytes = max_bytes
        self._wait_timeout_s = wait_timeout_s
        self._clock = clock
        self._sleep = sleep

    def export(self, camera_id: str, event_id: str, start: datetime, end: datetime) -> ClipResult:
        start, end = _utc(start), _utc(end)
        window_start, window_end = start - self._pre, end + self._post
        remaining_s = max(0.0, (window_end - self._clock()).total_seconds())
        wait_for_coverage(
            self._api_url,
            camera_id,
            window_end,
            timeout_s=remaining_s + self._wait_timeout_s,
            clock=self._clock,
            sleep=self._sleep,
        )
        dest = event_clip_path(self._events_dir, camera_id, start, event_id)
        return export_clip(
            self._api_url, camera_id, window_start, window_end, dest, max_bytes=self._max_bytes
        )
