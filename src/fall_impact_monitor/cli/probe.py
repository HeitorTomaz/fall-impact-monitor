"""`fim probe`: record per-frame timing of a file or stream."""

import argparse
import csv
import json
import time
from collections.abc import Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from fall_impact_monitor.capture.reader import (
    FrameEvent,
    ReconnectingReader,
    SourceOptions,
    StateEvent,
)
from fall_impact_monitor.capture.stats import summarize, to_markdown
from fall_impact_monitor.capture.timing import FrameRecord
from fall_impact_monitor.config import Settings, mask_url

_CSV_FIELDS = [
    "seq",
    "pts",
    "pts_s",
    "time_base",
    "arrival_ns",
    "wall_utc",
    "keyframe",
    "size_bytes",
    "quality",
]


def add_parser(subparsers: "argparse._SubParsersAction[argparse.ArgumentParser]") -> None:
    parser = subparsers.add_parser("probe", help="record per-frame timing of a file or stream")
    parser.add_argument("--source", required=True, help="file path or stream URL")
    parser.add_argument("--camera-id", default="probe")
    parser.add_argument(
        "--duration-s",
        type=float,
        default=60.0,
        help="stop after this many seconds (streams); files are read to the end",
    )
    parser.add_argument(
        "--video-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="RTSP: set up a video-only session (default) or all media",
    )
    parser.add_argument(
        "--wallclock",
        action="store_true",
        help="replace source timestamps with arrival wall-clock (FFmpeg option)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="output folder (default: artifacts/probe/<UTC timestamp>)",
    )
    parser.set_defaults(handler=run)


def run(args: argparse.Namespace) -> int:
    out_dir: Path = args.out_dir or Path("artifacts/probe") / datetime.now(UTC).strftime(
        "%Y%m%dT%H%M%SZ"
    )
    opts = SourceOptions(video_only=args.video_only, wallclock=args.wallclock)
    records: list[FrameRecord] = []
    states: list[StateEvent] = []
    reader = ReconnectingReader(
        args.source, args.camera_id, opts, stale_timeout_s=Settings().capture_stale_timeout_s
    )
    deadline = time.monotonic() + args.duration_s
    try:
        for event in reader:
            if isinstance(event, FrameEvent):
                records.append(event.record)
            else:
                states.append(event)
            if time.monotonic() >= deadline:
                break
    finally:
        reader.close()
        # Always keep what was captured, even on Ctrl+C or a late error.
        write_probe_outputs(out_dir, args.source, args.camera_id, opts, records, states)
    print(f"probe: {len(records)} frames from {mask_url(args.source)} -> {out_dir}")
    return 0 if records else 1


def write_probe_outputs(
    out_dir: Path,
    source: str,
    camera_id: str,
    opts: SourceOptions,
    records: Sequence[FrameRecord],
    states: Sequence[StateEvent],
) -> None:
    """Write frames.csv, summary.json and summary.md; credentials are always masked."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "frames.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        for r in records:
            writer.writerow(
                {
                    "seq": r.seq,
                    "pts": r.pts,
                    "pts_s": r.pts_s,
                    "time_base": str(r.time_base),
                    "arrival_ns": r.arrival_ns,
                    "wall_utc": r.wall_utc.isoformat(),
                    "keyframe": int(r.keyframe),
                    "size_bytes": r.size_bytes,
                    "quality": r.quality,
                }
            )
    timing = summarize(records) if records else None
    report = {
        "source": mask_url(source),
        "camera_id": camera_id,
        "options": asdict(opts),
        "generated_utc": datetime.now(UTC).isoformat(),
        "timing": asdict(timing) if timing else None,
        "states": [{"state": s.state, "detail": mask_url(s.detail)} for s in states],
    }
    (out_dir / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = [f"# Probe summary — `{mask_url(source)}`", "", f"Camera id: `{camera_id}`", ""]
    lines.append(to_markdown(timing) if timing else "No frames were received.\n")
    if states:
        lines += ["", "| Stream state | Detail |", "|---|---|"]
        lines += [f"| {s.state} | {mask_url(s.detail)} |" for s in states]
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
