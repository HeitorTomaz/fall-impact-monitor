"""Health observations: which conditions are active now, and the monitoring events that
report state changes.

Reasons are stable keys (``camera_lost``, ``camera_unstable``, ``analysis_slow``,
``processing_failing``, ``disk_low``, ``alerts_backlog``, ``database_backlog``,
``worker_stalled``); notification texts translate them.

Hysteresis: a camera enters "lost" after ``camera_lost_s`` without video, but once its
component is degraded any gap longer than ``camera_gap_s`` keeps it lost, so a camera
that drops for a minute and returns for a few seconds stays degraded instead of
producing a degraded/restored pair every cycle.
"""

import uuid
from dataclasses import dataclass
from datetime import datetime

from fall_impact_monitor.events import Event, EventCategory
from fall_impact_monitor.events.record import EventRecord
from fall_impact_monitor.health.monitor import Condition, Transition

SYSTEM = "system"
_NOTIFYING_PREFIXES = ("telegram:", "webhook")
# Video still flowing: a slow analysis rate is then the problem itself, not a symptom of a
# camera that stopped sending (which becomes camera_lost after its own limit).
_FRAMES_FLOWING_S = 2.0


@dataclass(frozen=True)
class HealthLimits:
    camera_lost_s: float = 30.0
    camera_gap_s: float = 10.0  # while degraded: a gap this long keeps the camera lost
    max_reconnects: int = 3  # within reconnect_window_s: an unstable camera
    reconnect_window_s: float = 600.0
    max_processing_failures: int = 5  # consecutive frames failing in the detectors
    min_free_bytes: int = 0
    alerts_max_age_s: float = 60.0  # the alert budget (rule 7)
    database_max_age_s: float = 300.0


@dataclass(frozen=True)
class CameraProbe:
    camera_id: str
    last_frame_s: float | None  # monotonic seconds; None before the first frame
    started_s: float
    rate_degraded: bool  # analysis below FIM_ANALYSIS_MIN_FPS
    busy: bool = False  # inside perception/detectors right now (frames are waiting)
    reconnects: int = 0  # within the reconnect window, the first connection excluded
    failures: int = 0  # consecutive frames that failed in perception or detectors


def observe(
    now_s: float,
    *,
    cameras: list[CameraProbe],
    limits: HealthLimits,
    degraded: set[str],
    disk_free_bytes: int,
    backlog_ages_s: dict[str, float | None],
) -> set[Condition]:
    found: set[Condition] = set()
    for camera in cameras:
        since = camera.last_frame_s if camera.last_frame_s is not None else camera.started_s
        age_s = now_s - since
        lost_after = limits.camera_gap_s if camera.camera_id in degraded else limits.camera_lost_s
        flowing = camera.busy or (camera.last_frame_s is not None and age_s <= _FRAMES_FLOWING_S)
        if not camera.busy and age_s > lost_after:
            found.add(Condition(camera.camera_id, "camera_lost"))
        elif flowing and camera.rate_degraded:
            found.add(Condition(camera.camera_id, "analysis_slow"))
        if camera.reconnects >= limits.max_reconnects:
            found.add(Condition(camera.camera_id, "camera_unstable"))
        if camera.failures >= limits.max_processing_failures:
            found.add(Condition(camera.camera_id, "processing_failing"))
    if disk_free_bytes < limits.min_free_bytes:
        found.add(Condition(SYSTEM, "disk_low"))
    for sink, age in backlog_ages_s.items():
        if age is None:
            continue
        if sink.startswith(_NOTIFYING_PREFIXES) and age > limits.alerts_max_age_s:
            found.add(Condition(SYSTEM, "alerts_backlog"))
        elif sink == "database" and age > limits.database_max_age_s:
            found.add(Condition(SYSTEM, "database_backlog"))
    return found


def health_record(transition: Transition, *, wall_utc: datetime, config_hash: str) -> EventRecord:
    """A monitoring event: never a near-miss, never aggregated (no track)."""
    category = (
        EventCategory.MONITORING_DEGRADED
        if transition.degraded
        else EventCategory.MONITORING_RESTORED
    )
    event = Event(
        event_id=str(uuid.uuid4()),
        camera_id=transition.component,
        category=category,
        confidence=1.0,
        start_s=0.0,
        end_s=0.0,
        track_id=None,
        identity="stable",
        near_miss=False,
        evidence={"reasons": list(transition.reasons)},
    )
    return EventRecord(event, wall_utc, wall_utc, wall_utc, config_hash)
