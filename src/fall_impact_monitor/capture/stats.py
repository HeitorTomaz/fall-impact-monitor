"""Timing summaries per time domain (source PTS and local arrival)."""

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise

from fall_impact_monitor.capture.timing import FrameRecord, TimestampQuality

_BUCKETS: tuple[tuple[str, float], ...] = (
    ("<30", 30.0),
    ("30-37", 37.0),
    ("37-45", 45.0),
    ("45-70", 70.0),
)


@dataclass(frozen=True)
class TimingSummary:
    frames: int
    duration_pts_s: float | None
    duration_arrival_s: float
    fps_pts: float | None
    fps_arrival: float
    quality_counts: dict[str, int]
    delta_ms_histogram: dict[str, int]
    keyframes: int
    mean_gop_s: float | None
    frames_per_second_min: int
    frames_per_second_max: int


def _bucket(delta_ms: float) -> str:
    if delta_ms <= 0:
        return "<=0"
    for name, upper in _BUCKETS:
        if delta_ms < upper:
            return name
    return ">=70"


def _rate(count: int, span_s: float) -> float | None:
    return (count - 1) / span_s if count > 1 and span_s > 0 else None


def summarize(records: Sequence[FrameRecord]) -> TimingSummary:
    """Summarize frame timing; PTS statistics ignore frames without PTS."""
    if not records:
        raise ValueError("no frame records to summarize")

    with_pts = [r.pts_s for r in records if r.pts_s is not None]
    duration_pts_s = with_pts[-1] - with_pts[0] if len(with_pts) > 1 else None
    histogram = dict.fromkeys(["<=0", *(name for name, _ in _BUCKETS), ">=70"], 0)
    for previous, current in pairwise(with_pts):
        histogram[_bucket((current - previous) * 1000.0)] += 1

    arrivals_s = [r.arrival_ns / 1e9 for r in records]
    duration_arrival_s = arrivals_s[-1] - arrivals_s[0]

    keyframe_times = [
        r.pts_s if r.pts_s is not None else r.arrival_ns / 1e9 for r in records if r.keyframe
    ]
    gops = [b - a for a, b in pairwise(keyframe_times)]

    per_second = Counter(int(t - arrivals_s[0]) for t in arrivals_s)
    full_seconds = [per_second[s] for s in range(int(duration_arrival_s))] or [len(records)]

    qualities = Counter(r.quality for r in records)
    return TimingSummary(
        frames=len(records),
        duration_pts_s=duration_pts_s,
        duration_arrival_s=duration_arrival_s,
        fps_pts=_rate(len(with_pts), duration_pts_s or 0.0),
        fps_arrival=_rate(len(records), duration_arrival_s) or 0.0,
        quality_counts={q.value: qualities.get(q, 0) for q in TimestampQuality},
        delta_ms_histogram=histogram,
        keyframes=sum(1 for r in records if r.keyframe),
        mean_gop_s=sum(gops) / len(gops) if gops else None,
        frames_per_second_min=min(full_seconds),
        frames_per_second_max=max(full_seconds),
    )


def _fmt(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def to_markdown(summary: TimingSummary) -> str:
    """Render a summary as a Markdown report."""
    lines = [
        "| Metric | Value |",
        "|---|---|",
        f"| Frames | {summary.frames} |",
        f"| Duration (source PTS) | {_fmt(summary.duration_pts_s)} s |",
        f"| Duration (arrival) | {_fmt(summary.duration_arrival_s)} s |",
        f"| FPS (source PTS) | {_fmt(summary.fps_pts, 2)} |",
        f"| FPS (arrival) | {_fmt(summary.fps_arrival, 2)} |",
        f"| Keyframes | {summary.keyframes} |",
        f"| Mean GOP | {_fmt(summary.mean_gop_s)} s |",
        f"| Frames per arrival second (min/max) | "
        f"{summary.frames_per_second_min}/{summary.frames_per_second_max} |",
        "",
        "| Timestamp quality | Frames |",
        "|---|---|",
        *(f"| {name} | {count} |" for name, count in summary.quality_counts.items()),
        "",
        "| PTS delta (ms) | Count |",
        "|---|---|",
        *(f"| {name} | {count} |" for name, count in summary.delta_ms_histogram.items()),
    ]
    return "\n".join(lines) + "\n"
