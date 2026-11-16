"""Event-level scores of the fall rules against ground-truth falls (#81).

A fall alert (``possible_fall`` or ``person_down``, not a near-miss) reports the instant at
which the fall was identified: its event start. A ground-truth fall is detected by an
alert whose instant lies from ``before_s`` before to ``after_start_s`` after the labelled
start of the fall and that was sent at most ``deadline_s`` after that start (an alert may
come late without harm; the maintainer's budget is 30 s, 60 s at most, reported apart).
Each fall takes one alert of its own, the one sent first; only then are the alerts whose
instant lies inside some fall's labelled range (``before_s`` before its start to
``after_s`` after its end) counted as duplicates (the service would aggregate them), so a
fall never loses its alert to a neighbour. Any other fall alert is false. Latency is the
time from the fall's start to the moment its alert was sent; it is negative when the
alert came before the labelled start. An alert sent after the clip's end was raised on
evaluation padding (``fim analyze --pad-end-s``) and is counted apart as well.

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
    deadline_s: float = 30.0
    late_deadline_s: float = 60.0
    detected_within_late_deadline: int = 0
    detected_on_padding: int = 0
    false_alerts_on_padding: int = 0


def _match(
    alerts: list[Detection],
    falls: list[TruthEvent],
    before_s: float,
    after_start_s: float,
    deadline_s: float,
) -> dict[int, int]:
    """Fall index -> index of its alert (the one sent first among the free ones)."""
    used: dict[int, int] = {}
    for f, fall in enumerate(falls):
        free = [
            i
            for i, a in enumerate(alerts)
            if fall.start_s - before_s <= a.start_s <= fall.start_s + after_start_s
            and a.emitted_s - fall.start_s <= deadline_s
            and i not in used.values()
        ]
        if free:
            used[f] = min(free, key=lambda i: alerts[i].emitted_s)
    return used


def score(
    results: list[VideoResult],
    *,
    dataset: str,
    before_s: float = 2.0,
    after_s: float = 2.0,
    after_start_s: float = 5.0,
    deadline_s: float = 30.0,
    late_deadline_s: float = 60.0,
) -> Report:
    detected = duplicates = false_alerts = late = padded_hits = padded_false = 0
    latencies: list[float] = []
    missed: set[str] = set()
    false: set[str] = set()
    for result in sorted(results, key=lambda r: r.name):
        alerts = sorted(
            (a for a in result.alerts if a.category in FALL_ALERTS), key=lambda a: a.start_s
        )
        falls = sorted(result.falls, key=lambda f: f.start_s)
        used = _match(alerts, falls, before_s, after_start_s, deadline_s)
        late += len(_match(alerts, falls, before_s, after_start_s, late_deadline_s))
        for f, fall in enumerate(falls):
            if f in used:
                hit = alerts[used[f]]
                detected += 1
                latencies.append(hit.emitted_s - fall.start_s)
                padded_hits += hit.emitted_s > result.duration_s
            else:
                missed.add(result.name)
        ranges = [(f.start_s - before_s, f.end_s + after_s) for f in falls]
        for i, a in enumerate(alerts):
            if i in used.values():
                continue
            if any(low <= a.start_s <= high for low, high in ranges):
                duplicates += 1
            else:
                false_alerts += 1
                padded_false += a.emitted_s > result.duration_s
                false.add(result.name)
    hours = sum(r.duration_s for r in results) / 3600
    falls_total = sum(len(r.falls) for r in results)
    return Report(
        dataset=dataset,
        videos=len(results),
        hours=hours,
        falls=falls_total,
        detected=detected,
        recall=detected / falls_total if falls_total else None,
        duplicates=duplicates,
        false_alerts=false_alerts,
        false_alerts_per_hour=false_alerts / hours if hours else None,
        latency_median_s=float(np.median(latencies)) if latencies else None,
        latency_p95_s=float(np.percentile(latencies, 95)) if latencies else None,
        latency_max_s=max(latencies) if latencies else None,
        missed=sorted(missed),
        false=sorted(false),
        deadline_s=deadline_s,
        late_deadline_s=late_deadline_s,
        detected_within_late_deadline=late,
        detected_on_padding=padded_hits,
        false_alerts_on_padding=padded_false,
    )
