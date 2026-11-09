"""`fim recordings`: maintenance of continuous recordings and event clips."""

import argparse
import asyncio
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

from fall_impact_monitor.config import Settings, load_cameras
from fall_impact_monitor.recording import Span, gaps, list_spans
from fall_impact_monitor.recording.retention import prune_clips, prune_recordings
from fall_impact_monitor.storage import make_engine, sync_spans, upgrade

_GB = 1_000_000_000


def add_parser(subparsers: "argparse._SubParsersAction[argparse.ArgumentParser]") -> None:
    parser = subparsers.add_parser("recordings", help="recording maintenance")
    commands = parser.add_subparsers(dest="recordings_command", required=True)
    retention = commands.add_parser(
        "retention", help="apply the disk cap to recordings and the age limit to event clips"
    )
    retention.add_argument("--dry-run", action="store_true", help="report without deleting")
    retention.set_defaults(handler=run_retention)
    index = commands.add_parser(
        "index", help="store the recorded spans of every camera and print recent gaps"
    )
    index.set_defaults(handler=run_index)


def _database_url() -> str:
    return Settings().database_url()


async def _store(url: str, listings: dict[str, list[Span]]) -> None:
    engine = make_engine(url)
    try:
        for camera_id, spans in listings.items():
            await sync_spans(engine, camera_id, spans)
    finally:
        await engine.dispose()


def run_index(args: argparse.Namespace) -> int:
    settings = Settings()
    url = _database_url()
    upgrade(url)
    listings = {
        camera.id: list_spans(settings.mediamtx_api_url, camera.id)
        for camera in load_cameras(os.environ)
    }
    asyncio.run(_store(url, listings))
    since = datetime.now(UTC) - timedelta(hours=24)
    for camera_id, spans in listings.items():
        recent = [s for s in spans if s.end > since]
        # Between recorded spans only: the segment being written ends "now".
        holes = gaps(recent, max(since, recent[0].start), recent[-1].end) if recent else []
        print(f"{camera_id}: {len(spans)} spans, {len(holes)} gaps in the last 24 h")
        for hole in holes:
            seconds = (hole.end - hole.start).total_seconds()
            print(f"  gap {hole.start.isoformat()} .. {hole.end.isoformat()} ({seconds:.0f} s)")
    return 0


def run_retention(args: argparse.Namespace) -> int:
    settings = Settings()
    report = prune_recordings(
        Path(settings.recordings_dir),
        max_bytes=int(settings.recordings_max_gb * _GB),
        min_free_bytes=int(settings.recordings_disk_min_free_gb * _GB),
        dry_run=args.dry_run,
    )
    clips = (
        0
        if args.dry_run
        else prune_clips(
            Path(settings.event_clip_dir),
            max_age_s=settings.event_clip_retention_days * 86400,
            now=datetime.now(UTC),
        )
    )
    print(
        f"recordings: {report.deleted_files} segments deleted "
        f"({report.freed_bytes / _GB:.2f} GB), {report.recordings_bytes / _GB:.2f} GB kept, "
        f"{report.free_bytes / _GB:.2f} GB free"
        f"{' - DISK PRESSURE' if report.pressure else ''}"
        f"{' (dry run)' if args.dry_run else ''}\n"
        f"event clips: {clips} deleted"
    )
    return 1 if report.pressure else 0
