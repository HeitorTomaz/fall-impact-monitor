"""`fim run`: the monitoring service (the `app` service in Docker Compose)."""

import argparse
import asyncio
import contextlib
import os
import signal
import sys

from fall_impact_monitor.config import CameraSettings, Settings, load_cameras
from fall_impact_monitor.config.logs import configure_logging
from fall_impact_monitor.service.app import run_service


def add_parser(subparsers: "argparse._SubParsersAction[argparse.ArgumentParser]") -> None:
    parser = subparsers.add_parser("run", help="run the monitoring service")
    parser.set_defaults(handler=run)


async def _main(settings: Settings, cameras: list[CameraSettings]) -> None:
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        with contextlib.suppress(NotImplementedError):  # not available on Windows
            loop.add_signal_handler(sig, stop.set)
    await run_service(settings, cameras, stop=stop)


def run(args: argparse.Namespace) -> int:
    cameras = load_cameras(os.environ)
    if not cameras:
        print("fim run: no cameras declared; set FIM_CAMERAS (see README)", file=sys.stderr)
        return 2
    settings = Settings()
    configure_logging(settings.log_level)
    asyncio.run(_main(settings, cameras))
    return 0
