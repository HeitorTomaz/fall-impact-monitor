"""`fim db`: database maintenance."""

import argparse

from fall_impact_monitor.config import Settings
from fall_impact_monitor.storage import upgrade


def add_parser(subparsers: "argparse._SubParsersAction[argparse.ArgumentParser]") -> None:
    parser = subparsers.add_parser("db", help="database maintenance")
    commands = parser.add_subparsers(dest="db_command", required=True)
    commands.add_parser("upgrade", help="apply pending migrations").set_defaults(handler=run)


def run(args: argparse.Namespace) -> int:
    upgrade(Settings().database_url())
    print("db: schema up to date")
    return 0
