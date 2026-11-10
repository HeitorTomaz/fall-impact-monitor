"""`fim eval`: event-level recall, false alerts per hour and alert latency on a public
dataset, with the current settings and models.

Each video is analysed with ``fim analyze`` (its outputs stay under ``--out-dir``); fall
alerts are matched to the dataset's ground truth (see ``evaluation/scoring.py``). The
datasets are downloaded by the installer and never redistributed; the report holds
counts and video names only. Measurements on stated datasets, never clinical accuracy.
"""

import argparse
import json
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

import av

from fall_impact_monitor.evaluation.scoring import Detection, Report, VideoResult, score
from fall_impact_monitor.evaluation.truth import (
    LabelledVideo,
    caucafall_videos,
    gmdcsa24_videos,
    urfd_videos,
)


def _fps(path: Path) -> float:
    with av.open(str(path)) as container:
        return float(container.streams.video[0].average_rate or 25)


_DATASETS: dict[str, Callable[[Path], list[LabelledVideo]]] = {
    "gmdcsa24": gmdcsa24_videos,
    "urfd": urfd_videos,
    "caucafall": lambda root: caucafall_videos(root, fps_of=_fps),
}


def add_parser(subparsers: "argparse._SubParsersAction[argparse.ArgumentParser]") -> None:
    parser = subparsers.add_parser("eval", help="event-level scores on a public dataset")
    parser.add_argument("--dataset", choices=sorted(_DATASETS), required=True)
    parser.add_argument("--root", type=Path, required=True, help="the dataset's folder")
    parser.add_argument("--out-dir", type=Path, help="default: artifacts/eval/<dataset>")
    parser.add_argument("--limit", type=int, default=0, help="evenly spaced subset; 0 = all")
    parser.add_argument("--manifest", type=Path, default=Path("models.toml"))
    parser.add_argument("--before-s", type=float, default=2.0, help="alert before a fall")
    parser.add_argument("--after-s", type=float, default=2.0, help="alert after a fall ends")
    parser.add_argument(
        "--reuse", action="store_true", help="score the analyses already in --out-dir again"
    )
    parser.set_defaults(handler=run)


def run_analysis(video: Path, out_dir: Path, manifest: Path) -> Path:
    """Run ``fim analyze`` on one video; returns its events.jsonl."""
    from fall_impact_monitor.cli.main import main

    code = main(
        [
            "analyze",
            "--source",
            str(video),
            "--out-dir",
            str(out_dir),
            "--manifest",
            str(manifest),
        ]
    )
    if code != 0:
        raise RuntimeError(f"analysis failed for {video.name} (exit {code})")
    return out_dir / "events.jsonl"


def _probe(path: Path) -> tuple[float, float]:
    """Media time of the first frame, and the duration, in seconds."""
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        start = 0.0
        for frame in container.decode(stream):
            start = float(frame.time or 0.0)
            break
        if container.duration:
            duration = container.duration / 1_000_000
        else:
            duration = float((stream.duration or 0) * (stream.time_base or 0))
    return start, duration


def _alerts(events: Path, start_s: float) -> tuple[Detection, ...]:
    alerts = []
    for line in events.read_text(encoding="utf-8").splitlines():
        event = json.loads(line)
        if event.get("near_miss"):
            continue
        emitted = event.get("emitted_s", event.get("end_s", event["start_s"]))
        alerts.append(Detection(event["category"], event["start_s"] - start_s, emitted - start_s))
    return tuple(alerts)


def _subset(videos: list[LabelledVideo], limit: int) -> list[LabelledVideo]:
    if limit <= 0 or limit >= len(videos):
        return videos
    step = len(videos) / limit
    return [videos[int(i * step)] for i in range(limit)]


def _markdown(report: Report) -> str:
    def seconds(value: float | None) -> str:
        return "n/a" if value is None else f"{value:.1f} s"

    rows = [
        ("Videos", f"{report.videos} ({report.hours * 60:.1f} min)"),
        ("Falls detected (recall)", f"{report.detected}/{report.falls} ({report.recall:.0%})"),
        ("False alerts", f"{report.false_alerts} ({report.false_alerts_per_hour:.1f} per hour)"),
        ("Duplicate alerts for a detected fall", str(report.duplicates)),
        (
            "Latency from the fall's start (median / p95 / max)",
            f"{seconds(report.latency_median_s)} / {seconds(report.latency_p95_s)} / "
            f"{seconds(report.latency_max_s)}",
        ),
    ]
    lines = [f"# {report.dataset}", "", "| Measure | Value |", "|---|---|"]
    lines += [f"| {name} | {value} |" for name, value in rows]
    lines += ["", "Missed falls: " + (", ".join(report.missed) or "none")]
    lines += ["", "Videos with false alerts: " + (", ".join(report.false) or "none"), ""]
    return "\n".join(lines)


def run(args: argparse.Namespace) -> int:
    out_dir = args.out_dir or Path("artifacts/eval") / args.dataset
    videos = _subset(_DATASETS[args.dataset](args.root), args.limit)
    if not videos:
        print(f"eval: no videos found for {args.dataset} in {args.root}")
        return 2
    results = []
    for index, video in enumerate(videos, 1):
        video_dir = out_dir / "videos" / f"{index:04d}"
        events = video_dir / "events.jsonl"
        if not (args.reuse and events.exists()):
            events = run_analysis(video.path, video_dir, args.manifest)
        start_s, duration_s = _probe(video.path)
        results.append(VideoResult(video.name, duration_s, video.falls, _alerts(events, start_s)))
        print(f"eval: {index}/{len(videos)} {video.name}")
    report = score(results, dataset=args.dataset, before_s=args.before_s, after_s=args.after_s)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    markdown = _markdown(report)
    (out_dir / "report.md").write_text(markdown, encoding="utf-8")
    print(markdown)
    return 0
