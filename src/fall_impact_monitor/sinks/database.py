"""The database as a sink: it receives every event (near-misses included), delivery
results, feedback and configuration snapshots through its own outbox queue, so the
alert path never waits for it and nothing is lost while it is down."""

from sqlalchemy.ext.asyncio import AsyncEngine

from fall_impact_monitor.sinks.base import Item
from fall_impact_monitor.storage.events import (
    save_config,
    save_event,
    save_feedback,
    save_notification,
)

_KINDS = ("event", "notification", "feedback", "config")


class DatabaseSink:
    name = "database"

    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine

    def accepts(self, item: Item) -> bool:
        return bool(item["kind"] in _KINDS)

    async def deliver(self, item: Item) -> None:
        async with self._engine.begin() as conn:
            kind = item["kind"]
            if kind == "event":
                await save_event(conn, item["record"])
            elif kind == "notification":
                await save_notification(conn, item)
            elif kind == "feedback":
                await save_feedback(conn, item)
            else:
                await save_config(conn, item)
