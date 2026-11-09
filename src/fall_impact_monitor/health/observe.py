"""Health observations: which conditions are active now, and the monitoring events that
report state changes.

Reasons are stable keys (``camera_lost``, ``analysis_slow``, ``disk_low``,
``alerts_backlog``, ``database_backlog``, ``worker_stalled``); notification texts
translate them.
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
class CameraProbe:
    camera_id: str
    last_frame_s: float | None  # monotonic seconds; None before the first frame
    started_s: float
    rate_degraded: bool  # analysis below FIM_ANALYSIS_MIN_FPS


def observe(
    now_s: float,
    *,
    cameras: list[CameraProbe],
    camera_lost_s: float,
    disk_free_bytes: int,
    min_free_bytes: int,
    backlog_ages_s: dict[str, float | None],
    backlog_max_age_s: float,
) -> set[Condition]:
    found: set[Condition] = set()
    for camera in cameras:
        since = camera.last_frame_s if camera.last_frame_s is not None else camera.started_s
        if now_s - since > camera_lost_s:
            found.add(Condition(camera.camera_id, "camera_lost"))
        elif (
            camera.last_frame_s is not None
            and now_s - camera.last_frame_s <= _FRAMES_FLOWING_S
            and camera.rate_degraded
        ):
            found.add(Condition(camera.camera_id, "analysis_slow"))
    if disk_free_bytes < min_free_bytes:
        found.add(Condition(SYSTEM, "disk_low"))
    for sink, age_s in backlog_ages_s.items():
        if age_s is None or age_s <= backlog_max_age_s:
            continue
        if sink.startswith(_NOTIFYING_PREFIXES):
            found.add(Condition(SYSTEM, "alerts_backlog"))
        elif sink == "database":
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
