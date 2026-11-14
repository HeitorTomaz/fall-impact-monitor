"""`fim windows`: labelled pose windows from the analyses ``fim eval`` saved, for training
and testing temporal classifiers (#15). Local files only; never publish them (they hold
pose traces of the people in the dataset)."""

import argparse
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from fall_impact_monitor.cli import evaluate
from fall_impact_monitor.evaluation.windows import (
    Window,
    group_of,
    labelled_windows,
    save_windows,
)


def add_parser(subparsers: "argparse._SubParsersAction[argparse.ArgumentParser]") -> None:
    parser = subparsers.add_parser("windows", help="labelled pose windows from saved analyses")
    parser.add_argument("--dataset", choices=sorted(evaluate._DATASETS), required=True)
    parser.add_argument("--root", type=Path, required=True, help="the dataset's folder")
    parser.add_argument("--out-dir", type=Path, required=True, help="the fim eval --out-dir")
    parser.add_argument("--output", type=Path, required=True, help=".npz file to write")
    parser.set_defaults(handler=run)


def _lines(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as lines:
        for line in lines:
            if line.strip():
                yield json.loads(line)


def run(args: argparse.Namespace) -> int:
    from fall_impact_monitor.cli.analyze import ANALYSIS_FORMAT

    videos = evaluate._DATASETS[args.dataset](args.root)
    windows: list[Window] = []
    groups: list[str] = []
    names: list[str] = []
    perception: object = None
    for video in videos:
        video_dir = evaluate._video_dir(args.out_dir, video)
        meta = evaluate._read_meta(video_dir)
        if (
            not isinstance(meta, dict)
            or not (video_dir / "complete").exists()
            or meta.get("format") != ANALYSIS_FORMAT
        ):
            print(f"windows: no usable analysis of {video.name}; run fim eval first")
            return 2
        if perception is None:
            perception = meta["perception"]
        elif meta["perception"] != perception:
            print(f"windows: {video.name} was analysed with another perception configuration")
            return 2
        start_s, _ = evaluate._probe(video.path, evaluate._time_scale(args.dataset, video.path))
        found = labelled_windows(_lines(video_dir / "tracks.jsonl"), video.falls, start_s)
        windows += found
        groups += [group_of(args.dataset, video.name)] * len(found)
        names += [video.name] * len(found)
    if not windows:
        print("windows: no windows (no tracks with keypoints)")
        return 2
    meta_out = {"dataset": args.dataset, "perception": perception, "format": ANALYSIS_FORMAT}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_windows(args.output, windows, groups, names, meta_out)
    falls = sum(w.label for w in windows)
    print(f"windows: {len(windows)} windows ({falls} during falls) -> {args.output}")
    return 0
