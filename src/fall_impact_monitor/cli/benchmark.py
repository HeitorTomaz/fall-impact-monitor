"""`fim benchmark detectors`: person-detector recall on fallen and upright people.

Uses locally downloaded CAUCAFall videos and labels (never redistributed).
"""

import argparse
import json
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

import av

from fall_impact_monitor.api.perception import Detector
from fall_impact_monitor.evaluation.caucafall import (
    load_caucafall_labels,
    match_recall,
    select_frames,
)
from fall_impact_monitor.perception.detector import load_detector
from fall_impact_monitor.perception.models import load_manifest, resolve

_VIDEO_SUFFIXES = {".avi", ".mp4"}


@dataclass(frozen=True)
class BenchmarkRow:
    model: str
    frames: int
    fall_boxes: int
    fall_recall: float
    other_boxes: int
    other_recall: float
    mean_ms: float


def _read_frames(video: Path, wanted: set[int]) -> dict[int, object]:
    frames: dict[int, object] = {}
    with av.open(str(video)) as container:
        for index, frame in enumerate(container.decode(video=0)):
            if index in wanted:
                frames[index] = frame.to_ndarray(format="bgr24")
            if len(frames) == len(wanted):
                break
    return frames


def run_detector_benchmark(
    dataset_root: Path,
    detectors: Mapping[str, Detector],
    frames_per_video: int,
    iou: float = 0.5,
) -> list[BenchmarkRow]:
    """Recall at ``iou`` on fall-labelled and other labelled people, plus mean latency."""
    totals = {name: [0, 0, 0, 0, 0, 0.0] for name in detectors}  # frames, fm, ft, om, ot, ms
    videos = sorted(p for p in dataset_root.rglob("*") if p.suffix.lower() in _VIDEO_SUFFIXES)
    for video in videos:
        with av.open(str(video)) as container:
            stream = container.streams.video[0]
            width, height = stream.codec_context.width, stream.codec_context.height
        labels = load_caucafall_labels(video, width=width, height=height)
        chosen = select_frames(labels, frames_per_video)
        frames = _read_frames(video, set(chosen))
        for index, image in frames.items():
            fall_truth = [b.box_xyxy for b in labels[index] if b.fall]
            other_truth = [b.box_xyxy for b in labels[index] if not b.fall]
            for name, detector in detectors.items():
                start = time.perf_counter()
                boxes = [d.box_xyxy for d in detector.detect(image)]  # type: ignore[arg-type]
                elapsed_ms = (time.perf_counter() - start) * 1000
                fall_hit, fall_total = match_recall(fall_truth, boxes, iou=iou)
                other_hit, other_total = match_recall(other_truth, boxes, iou=iou)
                t = totals[name]
                t[0] += 1
                t[1] += fall_hit
                t[2] += fall_total
                t[3] += other_hit
                t[4] += other_total
                t[5] += elapsed_ms
    return [
        BenchmarkRow(
            model=name,
            frames=int(t[0]),
            fall_boxes=int(t[2]),
            fall_recall=t[1] / t[2] if t[2] else 0.0,
            other_boxes=int(t[4]),
            other_recall=t[3] / t[4] if t[4] else 0.0,
            mean_ms=t[5] / t[0] if t[0] else 0.0,
        )
        for name, t in totals.items()
    ]


def add_parser(subparsers: "argparse._SubParsersAction[argparse.ArgumentParser]") -> None:
    parser = subparsers.add_parser("benchmark", help="benchmark perception models on local data")
    parser.add_argument("target", choices=["detectors"])
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="folder with CAUCAFall videos and per-frame labels",
    )
    parser.add_argument("--models", default="yolox_tiny,yolox_s")
    parser.add_argument("--frames-per-video", type=int, default=8)
    parser.add_argument("--manifest", type=Path, default=Path("models.toml"))
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/benchmark"))
    parser.set_defaults(handler=run)


def run(args: argparse.Namespace) -> int:
    manifest = load_manifest(args.manifest)
    detectors: dict[str, Detector] = {
        name: load_detector(resolve(manifest[name], args.models_dir))
        for name in (n.strip() for n in args.models.split(",") if n.strip())
    }
    rows = run_detector_benchmark(args.dataset_root, detectors, args.frames_per_video)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "detectors.json").write_text(
        json.dumps([asdict(r) for r in rows], indent=2), encoding="utf-8"
    )
    lines = [
        "| Model | Frames | Fall recall | Other recall | Mean ms/frame |",
        "|---|---|---|---|---|",
    ]
    lines += [
        f"| {r.model} | {r.frames} | {r.fall_recall:.3f} ({r.fall_boxes}) | "
        f"{r.other_recall:.3f} ({r.other_boxes}) | {r.mean_ms:.1f} |"
        for r in rows
    ]
    report = "\n".join(lines) + "\n"
    (args.out_dir / "detectors.md").write_text(report, encoding="utf-8")
    print(report)
    return 0
