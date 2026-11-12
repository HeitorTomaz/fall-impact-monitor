"""`fim` command-line interface."""

import argparse
import sys
from collections.abc import Callable

from fall_impact_monitor import __version__
from fall_impact_monitor.cli import (
    alert,
    analyze,
    benchmark,
    clip,
    datasets,
    db,
    evaluate,
    mediamtx_config,
    models,
    probe,
    recordings,
    run,
    zones,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fim", description="Fall Impact Monitor tools")
    parser.add_argument("--version", action="version", version=f"fim {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)
    probe.add_parser(subparsers)
    clip.add_parser(subparsers)
    models.add_parser(subparsers)
    benchmark.add_parser(subparsers)
    analyze.add_parser(subparsers)
    evaluate.add_parser(subparsers)
    datasets.add_parser(subparsers)
    zones.add_parser(subparsers)
    mediamtx_config.add_parser(subparsers)
    recordings.add_parser(subparsers)
    db.add_parser(subparsers)
    alert.add_parser(subparsers)
    run.add_parser(subparsers)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    handler: Callable[[argparse.Namespace], int] = args.handler
    return handler(args)


if __name__ == "__main__":
    sys.exit(main())
