"""`fim eval`: event-level recall, false alerts per hour and alert latency on a public
dataset, with the current settings and models.

Each video is analysed with ``fim analyze`` (its outputs stay under ``--out-dir``); fall
alerts are matched to the dataset's ground truth (see ``evaluation/scoring.py``). The
datasets are downloaded by the installer and never redistributed; the report holds
counts and video names only. Measurements on stated datasets, never clinical accuracy.

``--replay`` runs the current detectors again on the saved analyses, without perception,
and writes ``report.replay.*``: rule changes are measured in minutes (#67). It refuses
analyses saved with another perception configuration.
"""

import argparse
import json
import re
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

import av

from fall_impact_monitor.config import Settings
from fall_impact_monitor.evaluation.mcfd import MCFD_CAPTURE_FPS, mcfd_videos
from fall_impact_monitor.evaluation.scoring import Detection, Report, VideoResult, score
from fall_impact_monitor.evaluation.truth import (
    LabelledVideo,
    caucafall_videos,
    gmdcsa24_videos,
    urfd_videos,
)


def _fps(path: Path) -> float:
    """The video's frame rate; never assumed (ground truth is in frames)."""
    with av.open(str(path)) as container:
        rate = container.streams.video[0].average_rate
    if not rate:
        raise ValueError(f"{path.name}: no frame rate in the file")
    return float(rate)


# Settings that change what perception saves; a replay needs them unchanged. The detectors'
# own settings (thresholds, timings, which detectors run) are what a replay may change.
PERCEPTION_SETTINGS = (
    "person_detector_model",
    "person_detector_min_score",
    "person_detector_max_fps",
    "pose_estimator_model",
    "tracker_new_track_min_score",
    "tracker_lost_track_timeout_s",
    "analysis_empty_scene_fps",
    "analysis_person_visible_fps",
    "analysis_person_hold_s",
    "capture_max_analysis_width_px",
    "capture_packet_buffer_s",
    "fall_confirmation_enabled",
    "fall_confirmation_max_fps",
    "fall_confirmation_before_s",
    "fall_confirmation_after_s",
)


def perception_config(settings: Settings) -> dict[str, Any]:
    """The perception configuration, by environment variable, recorded with each analysis."""
    return {f"FIM_{name.upper()}": getattr(settings, name) for name in PERCEPTION_SETTINGS}


def video_meta(video: LabelledVideo, settings: Settings, time_scale: float) -> dict[str, Any]:
    from fall_impact_monitor.cli.analyze import ANALYSIS_FORMAT

    return {
        "video": video.name,
        "bytes": video.path.stat().st_size,
        "format": ANALYSIS_FORMAT,
        "perception": {**perception_config(settings), "media_time_scale": time_scale},
    }


_DATASETS: dict[str, Callable[[Path], list[LabelledVideo]]] = {
    "gmdcsa24": gmdcsa24_videos,
    "urfd": urfd_videos,
    "caucafall": lambda root: caucafall_videos(root, fps_of=_fps),
    "mcfd": lambda root: mcfd_videos(root, fps_of=lambda path: MCFD_CAPTURE_FPS),
}

# Media time multiplier for files whose declared frame rate is not the capture rate.
_TIME_SCALE: dict[str, Callable[[Path], float]] = {
    "mcfd": lambda path: _fps(path) / MCFD_CAPTURE_FPS,
}


def _time_scale(dataset: str, path: Path) -> float:
    scale = _TIME_SCALE.get(dataset)
    return scale(path) if scale else 1.0


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
    parser.add_argument(
        "--replay",
        action="store_true",
        help="run the current detectors again on the analyses in --out-dir, without perception",
    )
    parser.set_defaults(handler=run)


def run_analysis(video: Path, out_dir: Path, manifest: Path, time_scale: float = 1.0) -> Path:
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
            "--media-time-scale",
            repr(time_scale),
        ]
    )
    if code != 0:
        raise RuntimeError(f"analysis failed for {video.name} (exit {code})")
    return out_dir / "events.jsonl"


def _probe(path: Path, time_scale: float = 1.0) -> tuple[float, float]:
    """Media time of the first frame, and the duration, in (scaled) seconds."""
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
    return start * time_scale, duration * time_scale


def _read_meta(video_dir: Path) -> object:
    try:
        return json.loads((video_dir / "meta.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


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

    percent = "n/a" if report.recall is None else f"{report.recall:.0%}"
    rate = report.false_alerts_per_hour
    per_hour = "n/a" if rate is None else f"{rate:.1f}"
    rows = [
        ("Videos", f"{report.videos} ({report.hours * 60:.1f} min)"),
        ("Falls detected (recall)", f"{report.detected}/{report.falls} ({percent})"),
        ("False alerts", f"{report.false_alerts} ({per_hour} per hour)"),
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


def _video_dir(out_dir: Path, video: LabelledVideo) -> Path:
    return out_dir / "videos" / re.sub(r"[^A-Za-z0-9._-]+", "_", video.name)


def _replay_refusal(video_dir: Path, expected: dict[str, Any]) -> str | None:
    """Why a saved analysis cannot be replayed with the current settings, if it cannot."""
    meta = _read_meta(video_dir)
    if not (video_dir / "complete").exists() or not isinstance(meta, dict):
        return "no complete analysis saved; run fim eval without --replay first"
    if meta.get("format") != expected["format"] or "perception" not in meta:
        return "saved before replay was possible; run fim eval without --replay first"
    if {k: meta.get(k) for k in ("video", "bytes")} != {k: expected[k] for k in ("video", "bytes")}:
        return "saved for another video file; run fim eval without --replay first"
    changed = sorted(
        name
        for name in set(meta["perception"]) | set(expected["perception"])
        if meta["perception"].get(name) != expected["perception"].get(name)
    )
    if changed:
        return "analysed with another perception configuration: " + ", ".join(
            f"{name}={meta['perception'].get(name)} (now {expected['perception'].get(name)})"
            for name in changed
        )
    return None


def _replay(args: argparse.Namespace, videos: list[LabelledVideo], out_dir: Path) -> int:
    from fall_impact_monitor.cli.analyze import replay

    settings = Settings()
    for video in videos:
        expected = video_meta(video, settings, _time_scale(args.dataset, video.path))
        refusal = _replay_refusal(_video_dir(out_dir, video), expected)
        if refusal is not None:
            print(f"eval: cannot replay {video.name}: {refusal}")
            return 2
    results = []
    missing = 0
    for index, video in enumerate(videos, 1):
        video_dir = _video_dir(out_dir, video)
        replayed = replay(video_dir, settings)
        missing += replayed.missing_windows
        events = video_dir / "replay" / "events.jsonl"
        events.parent.mkdir(exist_ok=True)
        events.write_text("".join(json.dumps(e) + "\n" for e in replayed.events), encoding="utf-8")
        start_s, duration_s = _probe(video.path, _time_scale(args.dataset, video.path))
        results.append(VideoResult(video.name, duration_s, video.falls, _alerts(events, start_s)))
        print(f"eval: replayed {index}/{len(videos)} {video.name}")
    report = score(results, dataset=args.dataset, before_s=args.before_s, after_s=args.after_s)
    (out_dir / "report.replay.json").write_text(
        json.dumps(asdict(report), indent=2), encoding="utf-8"
    )
    markdown = _markdown(report)
    if missing:
        markdown += (
            f"\n{missing} fall near-misses were not analysed again at full rate: their "
            "windows are not in the saved analyses (the rules found new candidates). "
            "A full run measures them.\n"
        )
    (out_dir / "report.replay.md").write_text(markdown, encoding="utf-8")
    print(markdown)
    return 0


def run(args: argparse.Namespace) -> int:
    out_dir = args.out_dir or Path("artifacts/eval") / args.dataset
    videos = _subset(_DATASETS[args.dataset](args.root), args.limit)
    if not videos:
        print(f"eval: no videos found for {args.dataset} in {args.root}")
        return 2
    if args.replay:
        return _replay(args, videos, out_dir)
    settings = Settings()
    results = []
    for index, video in enumerate(videos, 1):
        video_dir = _video_dir(out_dir, video)
        events = video_dir / "events.jsonl"
        time_scale = _time_scale(args.dataset, video.path)
        meta = video_meta(video, settings, time_scale)
        done = video_dir / "complete"
        reusable = done.exists() and _read_meta(video_dir) == meta
        if not (args.reuse and reusable):
            done.unlink(missing_ok=True)
            events = run_analysis(video.path, video_dir, args.manifest, time_scale)
            (video_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
            done.touch()  # only after the analysis finished
        start_s, duration_s = _probe(video.path, time_scale)
        results.append(VideoResult(video.name, duration_s, video.falls, _alerts(events, start_s)))
        print(f"eval: {index}/{len(videos)} {video.name}")
    report = score(results, dataset=args.dataset, before_s=args.before_s, after_s=args.after_s)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    markdown = _markdown(report)
    (out_dir / "report.md").write_text(markdown, encoding="utf-8")
    print(markdown)
    return 0
