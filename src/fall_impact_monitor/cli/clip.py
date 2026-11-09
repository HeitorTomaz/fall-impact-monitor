"""`fim clip`: export a time range of a recorded path as an MP4 file."""

import argparse
from datetime import datetime
from pathlib import Path

from fall_impact_monitor.config import Settings
from fall_impact_monitor.recording import fetch_clip, playback_get_url


def add_parser(subparsers: "argparse._SubParsersAction[argparse.ArgumentParser]") -> None:
    parser = subparsers.add_parser("clip", help="export recorded video as an MP4 file")
    parser.add_argument("--camera", required=True, help="MediaMTX path name, e.g. cam1")
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
    api_url = args.api_url or Settings().mediamtx_api_url
    url = playback_get_url(api_url, args.camera, args.start, args.duration_s)
    dest = fetch_clip(url, args.out)
    print(f"clip: {dest} ({dest.stat().st_size} bytes)")
    return 0
