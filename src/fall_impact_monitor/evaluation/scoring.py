"""Event-level scores of the fall rules against ground-truth falls.

A ground-truth fall is detected when a fall alert (``possible_fall`` or ``person_down``,
not a near-miss) starts within ``before_s`` before the fall starts to ``after_s`` after it
ends. Each fall takes one alert of its own, the one raised first; only then are the alerts
left inside some fall's window counted as duplicates (the service would aggregate them),
so a fall never loses its alert to a neighbour's window. Any other fall alert is false.
Latency is the time from the fall's start to the moment its alert was raised; it is
negative when the alert came before the labelled start.

These are measurements on stated datasets, never clinical accuracy.
"""

from dataclasses import dataclass, field

import numpy as np

from fall_impact_monitor.evaluation.truth import TruthEvent

FALL_ALERTS = frozenset({"possible_fall", "person_down"})


@dataclass(frozen=True)
class Detection:
    category: str
    start_s: float  # start of the event's evidence, media seconds
    emitted_s: float  # when the event was raised, media seconds


@dataclass(frozen=True)
class VideoResult:
    name: str
    duration_s: float
    falls: tuple[TruthEvent, ...]
    alerts: tuple[Detection, ...]  # alerts only: near-misses are not notified


@dataclass(frozen=True)
class Report:
    dataset: str
    videos: int
    hours: float
    falls: int
    detected: int
    recall: float | None  # None without any fall (an ADL-only subset)
    duplicates: int
    false_alerts: int
    false_alerts_per_hour: float | None  # None when no duration could be read
    latency_median_s: float | None
    latency_p95_s: float | None
    latency_max_s: float | None
    missed: list[str] = field(default_factory=list)
    false: list[str] = field(default_factory=list)


def score(
    results: list[VideoResult], *, dataset: str, before_s: float = 2.0, after_s: float = 2.0
) -> Report:
    detected = duplicates = false_alerts = 0
    latencies: list[float] = []
    missed: set[str] = set()
    false: set[str] = set()
    for result in sorted(results, key=lambda r: r.name):
        alerts = sorted(
            (a for a in result.alerts if a.category in FALL_ALERTS), key=lambda a: a.start_s
        )
        windows = [
            (fall, fall.start_s - before_s, fall.end_s + after_s)
            for fall in sorted(result.falls, key=lambda f: f.start_s)
        ]
        used: set[int] = set()
        for fall, low, high in windows:
            free = [i for i, a in enumerate(alerts) if low <= a.start_s <= high and i not in used]
            if free:
                first = min(free, key=lambda i: alerts[i].emitted_s)
                used.add(first)
                detected += 1
                latencies.append(alerts[first].emitted_s - fall.start_s)
            else:
                missed.add(result.name)
        for i, a in enumerate(alerts):
            if i in used:
                continue
            if any(low <= a.start_s <= high for _, low, high in windows):
                duplicates += 1
            else:
                false_alerts += 1
                false.add(result.name)
    hours = sum(r.duration_s for r in results) / 3600
    falls = sum(len(r.falls) for r in results)
    return Report(
        dataset=dataset,
        videos=len(results),
        hours=hours,
        falls=falls,
        detected=detected,
        recall=detected / falls if falls else None,
        duplicates=duplicates,
        false_alerts=false_alerts,
        false_alerts_per_hour=false_alerts / hours if hours else None,
        latency_median_s=float(np.median(latencies)) if latencies else None,
        latency_p95_s=float(np.percentile(latencies, 95)) if latencies else None,
        latency_max_s=max(latencies) if latencies else None,
        missed=sorted(missed),
        false=sorted(false),
    )
