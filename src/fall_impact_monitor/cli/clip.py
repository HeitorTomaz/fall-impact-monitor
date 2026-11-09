"""`fim clip`: export a recorded time range of a camera as an MP4 file with a JSON sidecar
holding the requested and the actual windows."""

import argparse
from datetime import datetime, timedelta
from pathlib import Path

from fall_impact_monitor.config import Settings
from fall_impact_monitor.recording.clips import export_clip


def add_parser(subparsers: "argparse._SubParsersAction[argparse.ArgumentParser]") -> None:
    parser = subparsers.add_parser("clip", help="export recorded video as an MP4 file")
    parser.add_argument("--camera", required=True, help="camera id (its MediaMTX path), e.g. cam1")
    parser.add_argument(
        "--start",
        required=True,
        type=datetime.fromisoformat,
        help="ISO-8601 start with timezone, e.g. 2026-11-09T03:00:00Z",
    )
    parser.add_argument("--duration-s", required=True, type=float)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--api-url", default=None, help="MediaMTX playback URL (default: FIM_MEDIAMTX_API_URL)"
    )
    parser.set_defaults(handler=run)


def run(args: argparse.Namespace) -> int:
    settings = Settings()
    end = args.start + timedelta(seconds=args.duration_s)
    result = export_clip(
        args.api_url or settings.mediamtx_api_url,
        args.camera,
        args.start,
        end,
        args.out,
        max_bytes=int(settings.event_clip_max_mb * 1_000_000),
    )
    print(
        f"clip: {result.path} ({result.size_bytes} bytes"
        f"{', re-encoded' if result.transcoded else ''})\n"
        f"  requested {result.requested_start.isoformat()} .. {result.requested_end.isoformat()}\n"
        f"  actual    {result.actual_start.isoformat()} .. {result.actual_end.isoformat()}\n"
        f"  gaps: {len(result.gaps)}"
    )
    return 0
