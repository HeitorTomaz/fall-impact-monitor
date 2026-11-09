"""`fim recordings`: maintenance of continuous recordings and event clips."""

import argparse
from datetime import UTC, datetime
from pathlib import Path

from fall_impact_monitor.config import Settings
from fall_impact_monitor.recording.retention import prune_clips, prune_recordings

_GB = 1_000_000_000


def add_parser(subparsers: "argparse._SubParsersAction[argparse.ArgumentParser]") -> None:
    parser = subparsers.add_parser("recordings", help="recording maintenance")
    commands = parser.add_subparsers(dest="recordings_command", required=True)
    retention = commands.add_parser(
        "retention", help="apply the disk cap to recordings and the age limit to event clips"
    )
    retention.add_argument("--dry-run", action="store_true", help="report without deleting")
    retention.set_defaults(handler=run_retention)


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
