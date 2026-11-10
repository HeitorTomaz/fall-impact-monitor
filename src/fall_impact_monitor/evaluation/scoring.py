"""Event-level scores of the fall rules against ground-truth falls.

A ground-truth fall is detected when a fall alert (``possible_fall`` or ``person_down``,
not a near-miss) starts within ``before_s`` before the fall starts to ``after_s`` after it
ends. Each fall is matched once; further alerts for the same fall are duplicates (the
service would aggregate them), not false alerts. Any other fall alert is false. Latency is
the time from the fall's start to the moment the alert was raised.

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
    recall: float
    duplicates: int
    false_alerts: int
    false_alerts_per_hour: float
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
    missed: list[str] = []
    false: set[str] = set()
    for result in sorted(results, key=lambda r: r.name):
        alerts = sorted(
            (a for a in result.alerts if a.category in FALL_ALERTS), key=lambda a: a.start_s
        )
        used: set[int] = set()
        for fall in sorted(result.falls, key=lambda f: f.start_s):
            window = (fall.start_s - before_s, fall.end_s + after_s)
            hits = [i for i, a in enumerate(alerts) if window[0] <= a.start_s <= window[1]]
            free = [i for i in hits if i not in used]
            if free:
                used.add(free[0])
                detected += 1
                latencies.append(alerts[free[0]].emitted_s - fall.start_s)
                duplicates += len(free) - 1
                used.update(free[1:])
            else:
                missed.append(result.name)
        unmatched = len(alerts) - len(used)
        if unmatched:
            false_alerts += unmatched
            false.add(result.name)
    hours = sum(r.duration_s for r in results) / 3600
    falls = sum(len(r.falls) for r in results)
    return Report(
        dataset=dataset,
        videos=len(results),
        hours=hours,
        falls=falls,
        detected=detected,
        recall=detected / falls if falls else 0.0,
        duplicates=duplicates,
        false_alerts=false_alerts,
        false_alerts_per_hour=false_alerts / hours if hours else 0.0,
        latency_median_s=float(np.median(latencies)) if latencies else None,
        latency_p95_s=float(np.percentile(latencies, 95)) if latencies else None,
        latency_max_s=max(latencies) if latencies else None,
        missed=missed,
        false=sorted(false),
    )
