"""`fim alert test`: send a test message through every enabled output and report."""

import argparse
import asyncio
import tempfile
from pathlib import Path

from fall_impact_monitor.config import Settings
from fall_impact_monitor.sinks.factory import build_dispatcher


def add_parser(subparsers: "argparse._SubParsersAction[argparse.ArgumentParser]") -> None:
    parser = subparsers.add_parser("alert", help="alert outputs")
    commands = parser.add_subparsers(dest="alert_command", required=True)
    test = commands.add_parser("test", help="send a test message through every enabled output")
    test.add_argument("--timeout-s", type=float, default=30.0)
    test.set_defaults(handler=run_test)


async def _test(settings: Settings, timeout_s: float) -> dict[str, int]:
    # A private outbox: the running service keeps delivering its own queue.
    scratch = tempfile.TemporaryDirectory(prefix="fim-alert-test-")
    dispatcher, outputs = build_dispatcher(settings, engine=None, outbox_dir=Path(scratch.name))
    task = asyncio.create_task(dispatcher.run())
    await asyncio.sleep(0)
    dispatcher.submit_item({"kind": "test"})
    try:
        for _ in range(int(timeout_s * 10)):
            if not any(dispatcher.pending_counts().values()):
                break
            await asyncio.sleep(0.1)
        return dispatcher.pending_counts()
    finally:
        task.cancel()
        scratch.cleanup()
        if outputs.bot is not None:
            await outputs.bot.session.close()


def run_test(args: argparse.Namespace) -> int:
    pending = asyncio.run(_test(Settings(), args.timeout_s))
    for name, count in pending.items():
        print(f"{name}: {'delivered' if count == 0 else f'{count} pending (will retry)'}")
    return 0 if not any(pending.values()) else 1
