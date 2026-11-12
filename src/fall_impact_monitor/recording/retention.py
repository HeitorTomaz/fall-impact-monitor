"""Retention: a disk cap and a free-space floor for continuous recordings, an age limit for
event clips.

MediaMTX already deletes segments older than ``FIM_RECORDINGS_MAX_AGE_H``; this module adds
the disk limits it does not have. Only MediaMTX segment files
(``<recordings>/<path>/<YYYY-MM-DD_HH-MM-SS-ffffff>_<±HHMM>.mp4``) are ever deleted, oldest
first across cameras, and the newest segment of each path (the one being written) never
is. Event clips are evidence: they are deleted by age only, never for disk pressure.
"""

import logging
import re
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

_SEGMENT = re.compile(r"^(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}-\d{6})_([+-]\d{4})\.mp4$")


log = logging.getLogger(__name__)
_FILESYSTEM_FOLDERS = frozenset({"lost+found"})  # on a disk mounted as the folder


@dataclass(frozen=True)
class RetentionReport:
    deleted_files: int
    freed_bytes: int
    recordings_bytes: int
    free_bytes: int
    pressure: bool  # the cap or the free-space floor could not be met
    skipped_folders: tuple[str, ...] = ()  # camera folders that could not be read


def _segment_time(path: Path) -> datetime | None:
    match = _SEGMENT.match(path.name)
    if match is None:
        return None
    return datetime.strptime(match[1] + match[2], "%Y-%m-%d_%H-%M-%S-%f%z")


def _segments(
    recordings_dir: Path,
) -> tuple[list[tuple[datetime, Path, int]], int, tuple[str, ...]]:
    """Deletable segments (oldest first), the size of all segments, and the folders that
    could not be read (their segments are neither counted nor deleted)."""
    deletable: list[tuple[datetime, Path, int]] = []
    total = 0
    skipped: list[str] = []
    for folder in sorted(p for p in recordings_dir.iterdir() if p.is_dir()):
        if folder.name in _FILESYSTEM_FOLDERS:
            continue
        found = []
        try:
            entries = list(folder.iterdir())
        except OSError as error:  # keep pruning the other cameras
            skipped.append(folder.name)
            log.warning(
                "recordings folder not readable, skipped",
                extra={"folder": folder.name, "error": type(error).__name__},
            )
            continue
        for path in entries:
            started = _segment_time(path)
            if started is not None and path.is_file():
                size = path.stat().st_size
                found.append((started, path, size))
                total += size
        found.sort()
        deletable += found[:-1]  # the newest one may still be written
    deletable.sort()
    return deletable, total, tuple(skipped)


def prune_recordings(
    recordings_dir: Path,
    *,
    max_bytes: int,
    min_free_bytes: int,
    free_bytes: Callable[[], int] | None = None,
    dry_run: bool = False,
) -> RetentionReport:
    """Delete the oldest segments until they total at most ``max_bytes`` and the disk has
    ``min_free_bytes`` free. ``dry_run`` reports what would be deleted."""
    if not recordings_dir.is_dir():  # nothing recorded yet (fresh installation)
        return RetentionReport(0, 0, 0, 0, pressure=False)
    free = free_bytes or (lambda: shutil.disk_usage(recordings_dir).free)
    deletable, total, skipped = _segments(recordings_dir)
    available = free()
    deleted = freed = 0
    for _, path, size in deletable:
        if total <= max_bytes and available >= min_free_bytes:
            break
        if not dry_run:
            path.unlink(missing_ok=True)
        deleted, freed, total = deleted + 1, freed + size, total - size
        available = available + size if dry_run else free()
    return RetentionReport(
        deleted_files=deleted,
        freed_bytes=freed,
        recordings_bytes=total,
        free_bytes=available,
        pressure=total > max_bytes or available < min_free_bytes,
        skipped_folders=skipped,
    )


def prune_clips(events_dir: Path, *, max_age_s: float, now: datetime) -> int:
    """Delete event clips (and their JSON sidecars) last modified more than
    ``max_age_s`` before ``now``. Returns the number of clips deleted."""
    if not events_dir.is_dir():
        return 0
    cutoff = (now - timedelta(seconds=max_age_s)).timestamp()
    deleted = 0
    for clip in events_dir.rglob("*.mp4"):
        if clip.stat().st_mtime < cutoff:
            clip.unlink(missing_ok=True)
            clip.with_suffix(".json").unlink(missing_ok=True)
            deleted += 1
    return deleted
