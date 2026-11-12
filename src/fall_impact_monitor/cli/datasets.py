"""`fim datasets`: list the evaluation datasets and their terms, and fetch them.

Downloads are user-initiated, go outside the repository (or under the git-ignored
test_videos/) and are verified against the SHA-256 in datasets.toml.
"""

import argparse
from pathlib import Path

from fall_impact_monitor.evaluation import datasets


def add_parser(subparsers: "argparse._SubParsersAction[argparse.ArgumentParser]") -> None:
    parser = subparsers.add_parser("datasets", help="list or fetch evaluation datasets")
    parser.add_argument("action", choices=["list", "fetch"])
    parser.add_argument("name", nargs="?", help="dataset to fetch")
    parser.add_argument("--dest", type=Path, help="fetch: folder to write the dataset into")
    parser.add_argument("--manifest", type=Path, default=Path("datasets.toml"))
    parser.set_defaults(handler=run)


def _repo_root(start: Path) -> Path:
    for folder in (start, *start.parents):
        if (folder / "pyproject.toml").exists():
            return folder
    return start


def run(args: argparse.Namespace) -> int:
    manifest = datasets.load_dataset_manifest(args.manifest)
    if args.action == "list":
        for entry in manifest.values():
            print(
                f"{entry.name}: {entry.title} (v{entry.version}) — {entry.license}, "
                f"{entry.status}; {entry.source_url}"
            )
        return 0
    if args.name not in manifest or args.dest is None:
        print("datasets: fetch needs a known dataset name and --dest")
        return 2
    entry = manifest[args.name]
    print(f"datasets: {entry.name}: {entry.license} — cite: {entry.attribution}")
    try:
        dest = datasets.fetch_dataset(
            entry,
            args.dest,
            repo_root=_repo_root(Path.cwd()),
            downloader=datasets._urlretrieve,
        )
    except (datasets.DatasetError, OSError) as error:
        print(f"datasets: {entry.name}: {error}")
        return 1
    print(f"datasets: {entry.name}: {len(entry.files)} file(s) verified in {dest}")
    return 0
