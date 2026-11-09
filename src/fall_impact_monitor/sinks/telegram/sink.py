"""One Telegram chat as a sink (one outbox queue per chat, so a retry never re-sends to a
chat that already got the alert).

Aggregation (ADR-0006): a repeat of the same category by the same person with a stable
identity (same camera and track) within the window updates the existing message, which
Telegram does silently, until someone presses Seen; after that, a repeat is a new reply
that notifies, so a second event is never silent. Uncertain identities and other people
always get a new message. The aggregation state lives in memory: after a restart the
next event starts a new message.
"""

import asyncio
import hashlib
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime

from fall_impact_monitor.events.record import EventRecord
from fall_impact_monitor.events.wording import (
    Language,
    alert_text,
    button_labels,
    delivery_test_text,
)
from fall_impact_monitor.sinks.base import Item, is_notifiable
from fall_impact_monitor.sinks.telegram.api import ACTIONS, Button, MessageGone, TelegramApi

_Key = tuple[str, str, int]


@dataclass
class _Thread:
    message_id: int
    last_utc: datetime
    repeats: int = 0
    seen: bool = False


class TelegramChatSink:
    def __init__(
        self,
        api: TelegramApi,
        chat_id: int,
        *,
        language: Language,
        timezone: str,
        aggregation_window_s: float,
        late_after_s: float,
        report: Callable[[Item], None],
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
        min_interval_s: float = 1.0,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        # Chat ids are private: names (logs, outbox folders) carry a short hash instead.
        self.name = f"telegram:{hashlib.sha256(str(chat_id).encode()).hexdigest()[:8]}"
        self._api = api
        self._chat_id = chat_id
        self._language: Language = language
        self._timezone = timezone
        self._window_s = aggregation_window_s
        self._late_after_s = late_after_s
        self._report = report
        self._clock = clock
        self._min_interval_s = min_interval_s
        self._sleep = sleep
        self._monotonic = monotonic
        self._last_send: float | None = None
        self._threads: dict[_Key, _Thread] = {}

    def accepts(self, item: Item) -> bool:
        return is_notifiable(item)

    def mark_seen(self, message_id: int) -> None:
        for thread in self._threads.values():
            if thread.message_id == message_id:
                thread.seen = True

    def is_seen(self, message_id: int) -> bool:
        return any(t.seen for t in self._threads.values() if t.message_id == message_id)

    async def _pace(self) -> None:
        """About one message per second per chat (Telegram limit)."""
        now = self._monotonic()
        if self._last_send is not None:
            wait = self._min_interval_s - (now - self._last_send)
            if wait > 0:
                await self._sleep(wait)
                now = self._monotonic()
        self._last_send = now

    def _buttons(self, event_id: str) -> list[Button]:
        labels = button_labels(self._language)
        return [Button(action, event_id, labels[action]) for action in ACTIONS]

    def _key(self, record: EventRecord) -> _Key | None:
        event = record.event
        if self._window_s <= 0 or event.identity != "stable" or event.track_id is None:
            return None
        return (event.camera_id, event.category.value, event.track_id)

    async def deliver(self, item: Item) -> None:
        if item["kind"] == "test":
            await self._pace()
            await self._api.send_message(
                self._chat_id,
                delivery_test_text(self._language),
                buttons=[],
                reply_to=None,
                silent=False,
            )
            return
        record = EventRecord.from_json(item["record"])
        now = self._clock()
        self._threads = {
            k: t
            for k, t in self._threads.items()
            if (now - t.last_utc).total_seconds() <= self._window_s
        }
        key = self._key(record)
        thread = self._threads.get(key) if key is not None else None
        buttons = self._buttons(record.event.event_id)
        await self._pace()
        message_id: int | None = None
        if thread is not None and not thread.seen:
            message_id = await self._edit(thread, record, now, buttons)
            if message_id is None and key is not None:  # deleted in the chat
                del self._threads[key]
                thread = None
        if message_id is None:
            reply_to = thread.message_id if thread is not None else None
            message_id = await self._api.send_message(
                self._chat_id,
                self._text(record, now, 0),
                buttons=buttons,
                reply_to=reply_to,
                silent=False,
            )
            if key is not None:
                self._threads[key] = _Thread(message_id, now)
        self._report(
            {
                "kind": "notification",
                "event_id": record.event.event_id,
                "sink": self.name,
                "message_ref": str(message_id),
                "sent_utc": now.isoformat(),
                "late_s": round(max(0.0, (now - record.ended_utc).total_seconds()), 3),
            }
        )

    async def _edit(
        self, thread: _Thread, record: EventRecord, now: datetime, buttons: list[Button]
    ) -> int | None:
        """Update an aggregated message silently; None when it no longer exists."""
        text = self._text(record, now, thread.repeats + 1)
        try:
            await self._api.edit_message(self._chat_id, thread.message_id, text, buttons=buttons)
        except MessageGone:
            return None
        thread.repeats += 1
        thread.last_utc = now
        return thread.message_id

    def _text(self, record: EventRecord, now: datetime, repeats: int) -> str:
        return alert_text(
            record,
            language=self._language,
            timezone=self._timezone,
            now_utc=now,
            late_after_s=self._late_after_s,
            repeats=repeats,
            last_utc=now if repeats else None,
        )
