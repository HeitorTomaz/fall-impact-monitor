"""`fim benchmark`: measurements on local data.

- ``detectors``: person-detector recall on fallen and upright people, on locally downloaded
  CAUCAFall videos and labels (never redistributed).
- ``pipeline``: what one camera costs on this machine (decoding every frame; detection,
  tracking and pose at the person-visible rate), the highest sustainable analysis rate,
  CPU and memory; once with the person detector on every analysed frame and once at
  ``FIM_PERSON_DETECTOR_MAX_FPS`` (#68).
"""

import argparse
import json
import resource
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from fractions import Fraction
from pathlib import Path
from typing import Protocol

import av
import cv2
import numpy as np
from numpy.typing import NDArray

from fall_impact_monitor.api.perception import Detector, FrameResult
from fall_impact_monitor.capture import FrameRecord, TimestampQuality
from fall_impact_monitor.config import Settings
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


@dataclass(frozen=True)
class PipelineReport:
    frames: int
    frame_size: tuple[int, int]
    source_fps: float
    people_mean: float
    decode_ms_mean: float
    decode_ms_p95: float
    analysis_ms_mean: float  # detection, tracking and pose of one frame
    analysis_ms_p95: float
    max_analysis_fps: float  # decoding and analysing every frame
    full_rate_keeps_up: bool  # max_analysis_fps >= source_fps
    cameras_estimate: float  # cameras decoded at source_fps and analysed at person_fps
    decode_load: float  # seconds of decoding and conversion per second of video (#42)
    cpu_percent: float  # process CPU time / wall time (above 100 with several threads)
    max_rss_mb: float


def _p95(values: list[float]) -> float:
    return float(np.percentile(values, 95)) if values else 0.0


def summarize_pipeline(
    *,
    decode_ms: list[float],
    analysis_ms: list[float],
    people: list[int],
    source_fps: float,
    person_fps: float,
    cpu_s: float,
    wall_s: float,
    max_rss_mb: float,
    frame_size: tuple[int, int],
) -> PipelineReport:
    decode_mean = float(np.mean(decode_ms)) if decode_ms else 0.0
    analysis_mean = float(np.mean(analysis_ms)) if analysis_ms else 0.0
    per_frame_ms = decode_mean + analysis_mean
    max_fps = 1000.0 / per_frame_ms if per_frame_ms > 0 else 0.0
    per_camera_ms = decode_mean * source_fps + analysis_mean * person_fps
    return PipelineReport(
        frames=len(analysis_ms),
        frame_size=frame_size,
        source_fps=source_fps,
        people_mean=float(np.mean(people)) if people else 0.0,
        decode_ms_mean=decode_mean,
        decode_ms_p95=_p95(decode_ms),
        analysis_ms_mean=analysis_mean,
        analysis_ms_p95=_p95(analysis_ms),
        max_analysis_fps=max_fps,
        full_rate_keeps_up=max_fps >= source_fps,
        cameras_estimate=1000.0 / per_camera_ms if per_camera_ms > 0 else 0.0,
        decode_load=decode_mean * source_fps / 1000.0,
        cpu_percent=100.0 * cpu_s / wall_s if wall_s > 0 else 0.0,
        max_rss_mb=max_rss_mb,
    )


class _Pipeline(Protocol):
    def request_attention(self, until_s: float) -> None: ...
    def process(self, frame_bgr: NDArray[np.uint8], record: FrameRecord) -> FrameResult: ...


def run_pipeline_benchmark(
    source: str,
    pipeline: _Pipeline,
    *,
    max_frames: int,
    max_width_px: int,
    person_fps: float,
) -> PipelineReport:
    """Decode every frame of ``source`` and analyse frames ``1 / person_fps`` apart in
    video time, resized as the service does; ``max_frames`` analysed frames at most.
    Frames carry the video time as arrival time, so the detector's own rate applies as
    in the service."""
    decode_ms: list[float] = []
    analysis_ms: list[float] = []
    people: list[int] = []
    frame_size = (0, 0)
    wall_start, cpu_start = time.perf_counter(), time.process_time()
    with av.open(source) as container:
        stream = container.streams.video[0]
        source_fps = float(stream.average_rate or 0.0)
        rate = stream.time_base
        time_base = Fraction(rate.numerator, rate.denominator) if rate else Fraction(1)
        frames = container.decode(video=0)
        pipeline.request_attention(float("inf"))  # analyse every frame handed over
        interval_s = 1.0 / person_fps
        next_s = 0.0
        seq = -1
        while len(analysis_ms) < max_frames:
            seq += 1
            start = time.perf_counter()
            frame = next(frames, None)
            if frame is None:
                break
            image = np.asarray(frame.to_ndarray(format="bgr24"), dtype=np.uint8)
            height, width = image.shape[:2]
            if width > max_width_px:
                height, width = round(height * max_width_px / width), max_width_px
                image = np.asarray(
                    cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA), np.uint8
                )
            decode_ms.append((time.perf_counter() - start) * 1000)
            frame_size = (width, height)
            video_s = (
                float(frame.pts * time_base)
                if frame.pts is not None
                else seq / (source_fps or person_fps)
            )
            if video_s < next_s - 1e-6:
                continue
            next_s = video_s + interval_s
            record = FrameRecord(
                "benchmark",
                seq,
                frame.pts,
                time_base,
                int(video_s * 1e9),
                datetime.now(UTC),
                frame.key_frame,
                0,
                TimestampQuality.OK,
            )
            start = time.perf_counter()
            result = pipeline.process(image, record)
            analysis_ms.append((time.perf_counter() - start) * 1000)
            people.append(len(result.detections) if result.detections else len(result.tracks))
    max_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # KiB on Linux
    return summarize_pipeline(
        decode_ms=decode_ms,
        analysis_ms=analysis_ms,
        people=people,
        source_fps=source_fps,
        person_fps=person_fps,
        cpu_s=time.process_time() - cpu_start,
        wall_s=time.perf_counter() - wall_start,
        max_rss_mb=max_rss_mb,
        frame_size=frame_size,
    )


def add_parser(subparsers: "argparse._SubParsersAction[argparse.ArgumentParser]") -> None:
    parser = subparsers.add_parser("benchmark", help="measure models and the pipeline locally")
    parser.add_argument("target", choices=["detectors", "pipeline"])
    parser.add_argument(
        "--dataset-root",
        type=Path,
        help="detectors: folder with CAUCAFall videos and per-frame labels",
    )
    parser.add_argument("--models", default="yolox_tiny,yolox_s")
    parser.add_argument("--frames-per-video", type=int, default=8)
    parser.add_argument("--source", help="pipeline: a video file or stream with people in it")
    parser.add_argument("--frames", type=int, default=300, help="pipeline: frames to analyse")
    parser.add_argument("--manifest", type=Path, default=Path("models.toml"))
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/benchmark"))
    parser.set_defaults(handler=run)


def detector_rates(settings: Settings) -> list[float]:
    """Person-detector rates to compare: every analysed frame, then the configured one."""
    configured = settings.person_detector_max_fps
    return [0.0, configured] if configured > 0 else [0.0]


def _run_pipeline(args: argparse.Namespace) -> int:
    from fall_impact_monitor.api.build import build_pipeline
    from fall_impact_monitor.config import mask_url

    if not args.source:
        print("benchmark: pipeline needs --source")
        return 2
    settings = Settings()
    person_fps = settings.analysis_person_visible_fps
    runs = []
    for max_fps in detector_rates(settings):
        pipeline = build_pipeline(
            settings.model_copy(update={"person_detector_max_fps": max_fps}),
            started_s=time.monotonic(),
            manifest_path=args.manifest,
        )
        r = run_pipeline_benchmark(
            args.source,
            pipeline,
            max_frames=args.frames,
            max_width_px=settings.capture_max_analysis_width_px,
            person_fps=person_fps,
        )
        runs.append({"person_detector_max_fps": max_fps, **asdict(r)})
        detector = "every analysed frame" if max_fps == 0 else f"at most {max_fps:g} per second"
        print(
            "\n".join(
                [
                    f"person detector {detector}:",
                    f"  frames analysed: {r.frames} at {r.frame_size[0]}x{r.frame_size[1]}, "
                    f"{person_fps:g} per second of video, source {r.source_fps:.1f} fps, "
                    f"{r.people_mean:.1f} people per frame",
                    f"  decode: {r.decode_ms_mean:.1f} ms per frame (p95 {r.decode_ms_p95:.1f}), "
                    f"{r.decode_load:.2f} s per second of video (decode + conversion)",
                    f"  detection + tracking + pose: {r.analysis_ms_mean:.1f} ms per analysed "
                    f"frame (p95 {r.analysis_ms_p95:.1f})",
                    f"  highest analysis rate: {r.max_analysis_fps:.1f} fps; full rate keeps up: "
                    f"{'yes' if r.full_rate_keeps_up else 'no'}",
                    f"  cameras at {person_fps:g} fps with people (estimate): "
                    f"{r.cameras_estimate:.1f}",
                    f"  CPU {r.cpu_percent:.0f} % of one core, peak memory {r.max_rss_mb:.0f} MB",
                ]
            )
        )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "pipeline.json").write_text(
        json.dumps({"source": mask_url(args.source), "runs": runs}, indent=2), encoding="utf-8"
    )
    return 0


def run(args: argparse.Namespace) -> int:
    if args.target == "pipeline":
        return _run_pipeline(args)
    if args.dataset_root is None:
        print("benchmark: detectors needs --dataset-root")
        return 2
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
