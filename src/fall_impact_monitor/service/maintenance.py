"""Periodic maintenance: retention of recordings and clips, and the recorded-span index.

A failing step is logged and retried at the next interval; it never stops the service.
Disk pressure is logged here; reporting it as degraded monitoring belongs to health (#8).
"""

import asyncio
import logging
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncEngine

from fall_impact_monitor.config import CameraSettings, Settings
from fall_impact_monitor.recording import list_spans
from fall_impact_monitor.recording.retention import prune_clips, prune_recordings
from fall_impact_monitor.storage import sync_spans

log = logging.getLogger(__name__)
_GB = 1_000_000_000


async def maintenance_step(
    settings: Settings, *, cameras: list[CameraSettings], engine: AsyncEngine | None
) -> None:
    report = await asyncio.to_thread(
        prune_recordings,
        Path(settings.recordings_dir),
        max_bytes=int(settings.recordings_max_gb * _GB),
        min_free_bytes=int(settings.recordings_disk_min_free_gb * _GB),
    )
    if report.pressure:
        log.warning("disk pressure", extra={"free_bytes": report.free_bytes})
    await asyncio.to_thread(
        prune_clips,
        Path(settings.event_clip_dir),
        max_age_s=settings.event_clip_retention_days * 86400,
        now=datetime.now(UTC),
    )
    if engine is None:
        return
    for camera in cameras:
        try:
            spans = await asyncio.to_thread(list_spans, settings.mediamtx_playback_url, camera.id)
            await sync_spans(engine, camera.id, spans)
        except Exception as error:  # database starting or MediaMTX restarting: next interval
            log.warning(
                "span index not updated",
                extra={"camera": camera.id, "error": type(error).__name__},
            )


async def maintenance_loop(
    step: Callable[[], Awaitable[None]],
    *,
    interval_s: float,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> None:
    while True:
        try:
            await step()
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("maintenance step failed")
        await sleep(interval_s)
