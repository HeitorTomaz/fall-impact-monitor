import asyncio
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import text

from fall_impact_monitor.sinks.database import DatabaseSink
from fall_impact_monitor.sinks.dispatcher import Dispatcher
from fall_impact_monitor.sinks.event_log import JsonLogSink
from fall_impact_monitor.sinks.outbox import Outbox
from fall_impact_monitor.storage import make_engine
from tests.database import needs_database
from tests.sinks.helpers import make_record
from tests.sinks.test_dispatcher import RecordingSink, run_until

pytestmark = needs_database
NOW = "2026-11-09T03:00:05+00:00"


def deliver(url: str, items: list[dict[str, Any]]) -> None:
    async def main() -> None:
        engine = make_engine(url)
        sink = DatabaseSink(engine)
        try:
            for item in items:
                await sink.deliver(item)
        finally:
            await engine.dispose()

    asyncio.run(main())


def query(url: str, sql: str) -> list[tuple[Any, ...]]:
    async def main() -> list[tuple[Any, ...]]:
        engine = make_engine(url)
        try:
            async with engine.connect() as conn:
                return [tuple(row) for row in (await conn.execute(text(sql))).all()]
        finally:
            await engine.dispose()

    return asyncio.run(main())


def event_item(i: int = 0, **kw: Any) -> dict[str, Any]:
    return {"kind": "event", "record": make_record(i, **kw).to_json()}


def test_events_are_stored_once_with_their_configuration(database: str) -> None:
    item = event_item(0)
    deliver(database, [item, item])
    rows = query(database, "SELECT category, near_miss, state, config_hash, evidence FROM events")
    assert rows == [("possible_fall", False, "detected", "cfg", {"max_drop_heights": 0.6})]
    started = query(database, "SELECT started_utc FROM events")[0][0]
    assert started == datetime(2026, 11, 9, 3, 0, 0, tzinfo=UTC)


def test_notifications_and_feedback_update_the_event_state(database: str) -> None:
    item = event_item(0)
    event_id = item["record"]["event_id"]
    notification = {
        "kind": "notification",
        "event_id": event_id,
        "sink": "telegram:1",
        "message_ref": "42",
        "sent_utc": NOW,
        "late_s": 0.0,
    }
    deliver(database, [item, notification, notification])
    assert query(database, "SELECT state FROM events") == [("notified",)]
    assert query(database, "SELECT sink, message_ref FROM notifications") == [("telegram:1", "42")]
    seen = {"kind": "feedback", "event_id": event_id, "action": "seen", "created_utc": NOW}
    deliver(database, [seen])
    assert query(database, "SELECT state, feedback FROM events") == [("seen", None)]
    answer = {**seen, "action": "false_alarm"}
    deliver(database, [answer])
    assert query(database, "SELECT state, feedback FROM events") == [("feedback", "false_alarm")]
    assert query(database, "SELECT action FROM feedback ORDER BY id") == [
        ("seen",),
        ("false_alarm",),
    ]


def test_configuration_snapshots_are_stored_once(database: str) -> None:
    config = {"kind": "config", "config_hash": "cfg", "snapshot": {"settings": {"a": 1}}}
    deliver(database, [config, config])
    assert query(database, "SELECT config_hash, snapshot FROM config_snapshots") == [
        ("cfg", {"settings": {"a": 1}})
    ]


def test_near_misses_are_stored_too(database: str) -> None:
    deliver(database, [event_item(0, near_miss=True)])
    assert query(database, "SELECT near_miss FROM events") == [(True,)]


def test_alerts_are_delivered_while_the_database_is_down_and_stored_after(
    database: str, tmp_path: Path
) -> None:
    outbox = Outbox(tmp_path / "outbox")
    notifier = RecordingSink("notifier")

    async def no_wait(seconds: float) -> None:
        await asyncio.sleep(0)

    down = make_engine("postgresql+asyncpg://fim:fim@127.0.0.1:1/missing")
    first = Dispatcher(
        outbox,
        [JsonLogSink(tmp_path / "events.jsonl", late_after_s=60), notifier, DatabaseSink(down)],
        sleep=no_wait,
    )
    record = make_record(0)
    run_until(first, lambda: len(notifier.delivered) == 1, before=lambda: first.submit(record))
    assert first.pending_counts()["database"] == 1  # still queued, nothing lost
    assert (tmp_path / "events.jsonl").exists()

    async def recover() -> None:
        engine = make_engine(database)
        second = Dispatcher(outbox, [DatabaseSink(engine)])
        task = asyncio.create_task(second.run())
        for _ in range(500):
            if second.pending_counts()["database"] == 0:
                break
            await asyncio.sleep(0.01)
        task.cancel()
        await engine.dispose()

    asyncio.run(recover())
    assert query(database, "SELECT event_id::text FROM events") == [(record.event.event_id,)]
