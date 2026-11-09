"""Dispatcher: durable fan-out of items to sinks, one asyncio worker per sink.

``submit`` writes one outbox file per accepting sink and returns at once (it is safe to
call from capture threads); nothing on the alert path waits for a destination. Each
worker delivers its items in order and retries a failing item with exponential backoff,
so a slow or unreachable sink (the database, the internet) never delays the others.
"""

import asyncio
import logging
import threading
from collections import deque
from collections.abc import Awaitable, Callable

from fall_impact_monitor.events.record import EventRecord
from fall_impact_monitor.sinks.base import Item, RetryLater, Sink
from fall_impact_monitor.sinks.outbox import Outbox

log = logging.getLogger(__name__)
_ESCALATE_AFTER = 5  # failed attempts of one item before logging an error


class Dispatcher:
    def __init__(
        self,
        outbox: Outbox,
        sinks: list[Sink],
        *,
        backoff_initial_s: float = 1.0,
        backoff_max_s: float = 60.0,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._outbox = outbox
        self._sinks = list(sinks)
        self._backoff_initial_s = backoff_initial_s
        self._backoff_max_s = backoff_max_s
        self._sleep = sleep
        self._loop: asyncio.AbstractEventLoop | None = None
        self._wake: dict[str, asyncio.Event] = {}
        self._lock = threading.Lock()
        # Items whose outbox write failed (disk full): kept in memory, never dropped.
        self._memory: dict[str, deque[Item]] = {sink.name: deque() for sink in self._sinks}

    @property
    def sinks(self) -> list[Sink]:
        return list(self._sinks)

    def submit(self, record: EventRecord) -> None:
        self.submit_item({"kind": "event", "record": record.to_json()})

    def submit_item(self, item: Item) -> None:
        """Queue ``item`` for every sink that accepts it. Thread-safe, non-blocking."""
        for sink in self._sinks:
            if not sink.accepts(item):
                continue
            try:
                self._outbox.put(sink.name, item)
            except OSError as error:
                log.error(
                    "outbox write failed, item kept in memory",
                    extra={"sink": sink.name, "error": type(error).__name__},
                )
                self._memory[sink.name].append(item)
        with self._lock:
            loop = self._loop
        if loop is not None and not loop.is_closed():
            loop.call_soon_threadsafe(self._wake_all)

    def _wake_all(self) -> None:
        for event in self._wake.values():
            event.set()

    def pending_counts(self) -> dict[str, int]:
        return {
            sink.name: self._outbox.count(sink.name) + len(self._memory[sink.name])
            for sink in self._sinks
        }

    async def run(self) -> None:
        """Deliver forever (cancel to stop)."""
        with self._lock:
            self._loop = asyncio.get_running_loop()
        self._wake = {sink.name: asyncio.Event() for sink in self._sinks}
        try:
            await asyncio.gather(*(self._worker(sink) for sink in self._sinks))
        finally:
            with self._lock:
                self._loop = None

    async def _worker(self, sink: Sink) -> None:
        wake = self._wake[sink.name]
        memory = self._memory[sink.name]
        while True:
            try:
                wake.clear()
                while memory:
                    await self._deliver(sink, None, memory[0])
                    memory.popleft()
                items = self._outbox.pending(sink.name)
                if not items:
                    await wake.wait()
                    continue
                for item_id, item in items:
                    await self._deliver(sink, item_id, item)
            except asyncio.CancelledError:
                raise
            except Exception:  # an outbox error must never stop deliveries
                log.exception("delivery worker error, continuing", extra={"sink": sink.name})
                await self._sleep(self._backoff_initial_s)

    async def _deliver(self, sink: Sink, item_id: str | None, item: Item) -> None:
        delay_s = self._backoff_initial_s
        attempts = 0
        while True:
            try:
                await sink.deliver(item)
            except RetryLater as wait:
                await self._sleep(wait.seconds)
                continue
            except asyncio.CancelledError:
                raise
            except Exception as error:
                attempts += 1
                level = logging.ERROR if attempts == _ESCALATE_AFTER else logging.WARNING
                text = (
                    "delivery still failing" if attempts >= _ESCALATE_AFTER else "delivery failed"
                )
                log.log(
                    level,
                    f"{text}, retrying",
                    extra={
                        "sink": sink.name,
                        "error": type(error).__name__,
                        "detail": str(error)[:200],
                        "attempts": attempts,
                        "retry_s": delay_s,
                    },
                )
                await self._sleep(delay_s)
                delay_s = min(delay_s * 2, self._backoff_max_s)
                continue
            if item_id is not None:
                self._outbox.done(sink.name, item_id)
            return
