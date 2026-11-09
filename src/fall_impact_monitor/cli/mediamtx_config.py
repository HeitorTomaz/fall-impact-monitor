"""`fim mediamtx-config`: write the MediaMTX configuration from the camera variables."""

import argparse
import os
from pathlib import Path

from fall_impact_monitor.config import Settings, load_cameras
from fall_impact_monitor.recording.mediamtx_config import (
    build_mediamtx_config,
    write_mediamtx_config,
)


def add_parser(subparsers: "argparse._SubParsersAction[argparse.ArgumentParser]") -> None:
    parser = subparsers.add_parser(
        "mediamtx-config", help="write the MediaMTX configuration from FIM_* camera variables"
    )
    parser.add_argument("--out", required=True, type=Path)
    parser.set_defaults(handler=run)


def run(args: argparse.Namespace) -> int:
    cameras = load_cameras(os.environ)
    settings = Settings()
    write_mediamtx_config(
        build_mediamtx_config(cameras, max_age_h=settings.recording_max_age_h), args.out
    )
    print(f"mediamtx-config: {len(cameras)} cameras -> {args.out}")
    return 0
