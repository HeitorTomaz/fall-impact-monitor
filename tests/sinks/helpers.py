"""Synthetic event records for sink tests."""

from datetime import UTC, datetime, timedelta
from typing import Any

from fall_impact_monitor.events import EventCategory, make_event
from fall_impact_monitor.events.record import EventRecord

T0 = datetime(2026, 11, 9, 3, 0, 0, tzinfo=UTC)


def make_record(
    i: int = 0,
    *,
    near_miss: bool = False,
    category: EventCategory = EventCategory.POSSIBLE_FALL,
    track_id: int | None = 1,
    identity: str = "stable",
    evidence: dict[str, Any] | None = None,
    at: datetime | None = None,
) -> EventRecord:
    start = at or T0 + timedelta(seconds=10 * i)
    event = make_event(
        camera_id="cam1",
        category=category,
        confidence=0.3 if near_miss else 0.72,
        start_s=float(i),
        end_s=float(i) + 1.0,
        track_id=track_id,
        identity="uncertain" if identity == "uncertain" else "stable",
        evidence=evidence or {"max_drop_heights": 0.6},
        analysed_seqs=[i],
        threshold=0.5,
    )
    return EventRecord(
        event, start, start + timedelta(seconds=1), start + timedelta(seconds=2), "cfg"
    )
