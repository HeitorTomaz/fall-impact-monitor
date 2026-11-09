"""`fim models`: fetch and verify model weights listed in the manifest."""

import argparse
from pathlib import Path

from fall_impact_monitor.perception.models import ModelError, fetch, load_manifest, resolve


def add_parser(subparsers: "argparse._SubParsersAction[argparse.ArgumentParser]") -> None:
    parser = subparsers.add_parser("models", help="fetch or verify model weights")
    parser.add_argument("action", choices=["fetch", "verify"])
    parser.add_argument("--only", default="", help="comma-separated model names (default: all)")
    parser.add_argument("--manifest", type=Path, default=Path("models.toml"))
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.set_defaults(handler=run)


def run(args: argparse.Namespace) -> int:
    manifest = load_manifest(args.manifest)
    names = [n.strip() for n in args.only.split(",") if n.strip()] or list(manifest)
    unknown = [n for n in names if n not in manifest]
    if unknown:
        print(f"models: unknown model(s): {', '.join(unknown)}")
        return 2
    failures = 0
    for name in names:
        spec = manifest[name]
        try:
            if args.action == "fetch":
                path = fetch(spec, args.models_dir)
            else:
                path = resolve(spec, args.models_dir)
            print(f"models: {name}: ok ({path})")
        except (ModelError, OSError) as exc:
            failures += 1
            print(f"models: {name}: {exc}")
    return 1 if failures else 0
