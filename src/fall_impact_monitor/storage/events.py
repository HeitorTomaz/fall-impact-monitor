"""Persistence of events, notifications, feedback and configuration snapshots.

Every write is idempotent where a retry could repeat it (events, notifications,
snapshots), because the database sink retries until the database accepts a write.
"""

from datetime import datetime
from typing import Any

from sqlalchemy import update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncConnection

from fall_impact_monitor.storage.models import (
    ConfigSnapshotRow,
    EventRow,
    FeedbackRow,
    NotificationRow,
)


def _utc(value: str) -> datetime:
    return datetime.fromisoformat(value)


async def save_event(conn: AsyncConnection, record: dict[str, Any]) -> None:
    values = {
        "event_id": record["event_id"],
        "camera_id": record["camera_id"],
        "category": record["category"],
        "confidence": record["confidence"],
        "near_miss": record["near_miss"],
        "started_utc": _utc(record["started_utc"]),
        "ended_utc": _utc(record["ended_utc"]),
        "detected_utc": _utc(record["detected_utc"]),
        "track_id": record["track_id"],
        "identity": record["identity"],
        "evidence": record["evidence"],
        "analysed_seqs": record["analysed_seqs"],
        "config_hash": record["config_hash"],
    }
    await conn.execute(insert(EventRow).values(values).on_conflict_do_nothing())


async def save_notification(conn: AsyncConnection, item: dict[str, Any]) -> None:
    values = {
        "event_id": item["event_id"],
        "sink": item["sink"],
        "message_ref": item.get("message_ref"),
        "sent_utc": _utc(item["sent_utc"]),
        "late_s": item.get("late_s", 0.0),
    }
    stmt = insert(NotificationRow).values(values)
    stmt = stmt.on_conflict_do_update(
        index_elements=["event_id", "sink"],
        set_={"message_ref": stmt.excluded.message_ref, "sent_utc": stmt.excluded.sent_utc},
    )
    await conn.execute(stmt)
    await conn.execute(
        update(EventRow)
        .where(EventRow.event_id == item["event_id"], EventRow.state == "detected")
        .values(state="notified")
    )


async def save_feedback(conn: AsyncConnection, item: dict[str, Any]) -> None:
    event_id, action = item["event_id"], item["action"]
    await conn.execute(
        insert(FeedbackRow).values(
            event_id=event_id, action=action, created_utc=_utc(item["created_utc"])
        )
    )
    if action == "seen":
        stmt = (
            update(EventRow)
            .where(EventRow.event_id == event_id, EventRow.state.in_(("detected", "notified")))
            .values(state="seen")
        )
    else:
        stmt = (
            update(EventRow)
            .where(EventRow.event_id == event_id)
            .values(state="feedback", feedback=action)
        )
    await conn.execute(stmt)


async def save_config(conn: AsyncConnection, item: dict[str, Any]) -> None:
    await conn.execute(
        insert(ConfigSnapshotRow)
        .values(config_hash=item["config_hash"], snapshot=item["snapshot"])
        .on_conflict_do_nothing()
    )
