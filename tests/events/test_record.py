from datetime import UTC, datetime, timedelta

import pytest

from fall_impact_monitor.events import EventCategory, make_event
from fall_impact_monitor.events.record import EventRecord

T0 = datetime(2026, 11, 9, 3, 0, 0, tzinfo=UTC)


def record(**changes: object) -> EventRecord:
    event = make_event(
        camera_id="cam1",
        category=EventCategory.POSSIBLE_FALL,
        confidence=0.72,
        start_s=100.0,
        end_s=101.5,
        track_id=3,
        identity="stable",
        evidence={"max_drop_heights": 0.6},
        analysed_seqs=[1, 2, 3],
        threshold=0.5,
    )
    fields: dict[str, object] = {
        "event": event,
        "started_utc": T0,
        "ended_utc": T0 + timedelta(seconds=1.5),
        "detected_utc": T0 + timedelta(seconds=2),
        "config_hash": "abc123",
    }
    fields.update(changes)
    return EventRecord(**fields)  # type: ignore[arg-type]


def test_json_round_trip() -> None:
    original = record()
    data = original.to_json()
    assert data["event_id"] == original.event.event_id
    assert data["category"] == "possible_fall"
    assert data["started_utc"] == "2026-11-09T03:00:00+00:00"
    assert EventRecord.from_json(data) == original


def test_times_must_be_timezone_aware() -> None:
    with pytest.raises(ValueError, match="timezone"):
        record(started_utc=datetime(2026, 11, 9))
