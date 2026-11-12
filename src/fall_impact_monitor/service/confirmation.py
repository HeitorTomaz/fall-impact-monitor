"""Full-rate confirmation of fall candidates, in its own thread (see
``api/confirmation.py``). Camera workers hand near-miss records over without ever
waiting: a full queue refuses the request (the near-miss itself is already recorded).
The worker waits until the packet buffer holds the frames ``after_s`` past the
candidate (or a grace period more, for a camera that stopped), decodes the window
from ``before_s`` before it, and sends a confirmed fall like any other event."""

import contextlib
import logging
import queue
import threading
import time
from collections.abc import Callable, Iterable, Mapping
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol

from fall_impact_monitor.api.confirmation import is_candidate
from fall_impact_monitor.events import Event
from fall_impact_monitor.events.record import EventRecord

log = logging.getLogger(__name__)
_GRACE_S = 5.0  # beyond after_s, for a camera that stopped streaming
_POLL_S = 0.5


class _Buffer(Protocol):
    def newest_arrival_ns(self) -> int | None: ...

    def decode_window(self, start_ns: int, end_ns: int) -> Iterable[Any]: ...


class _Confirmer(Protocol):
    def confirm(self, frames: Iterable[Any]) -> list[Event]: ...


class ConfirmationWorker:
    def __init__(
        self,
        confirmer_for: Callable[[str], _Confirmer],
        buffers: Mapping[str, _Buffer],
        on_event: Callable[[EventRecord], None],
        *,
        before_s: float,
        after_s: float,
        max_queue: int = 2,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._confirmer_for = confirmer_for
        self._buffers = buffers
        self._on_event = on_event
        self._before_s = before_s
        self._after_s = after_s
        self._clock = clock
        self._sleep = sleep
        self._queue: queue.Queue[EventRecord | None] = queue.Queue(maxsize=max_queue)
        self._thread: threading.Thread | None = None

    def request(self, record: EventRecord) -> bool:
        """Queue a candidate; never blocks. False when it is not one or the queue is full."""
        event = record.event
        if not is_candidate(event) or event.camera_id not in self._buffers:
            return False
        try:
            self._queue.put_nowait(record)
        except queue.Full:
            log.warning(
                "confirmation queue full, candidate not confirmed", extra={"event": event.event_id}
            )
            return False
        return True

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="confirmation", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._thread is not None:
            with contextlib.suppress(queue.Full):  # full: the daemon thread ends with us
                self._queue.put_nowait(None)
            self._thread.join(timeout=5)

    def _run(self) -> None:
        while self.process_next(timeout_s=None):
            pass

    def process_next(self, timeout_s: float | None) -> bool:
        """Confirm one queued candidate; False when stopped (or nothing came in time)."""
        try:
            record = self._queue.get(timeout=timeout_s) if timeout_s else self._queue.get_nowait()
        except queue.Empty:
            return False
        if record is None:
            return False
        try:
            self._confirm(record)
        except Exception:
            log.exception("confirmation failed", extra={"event": record.event.event_id})
        return True

    def _confirm(self, candidate: EventRecord) -> None:
        event = candidate.event
        buffer = self._buffers[event.camera_id]
        end_ns = int((event.end_s + self._after_s) * 1e9)
        waited_s = 0.0
        while waited_s < self._after_s + _GRACE_S:
            newest = buffer.newest_arrival_ns()
            if newest is not None and newest >= end_ns:
                break
            self._sleep(_POLL_S)
            waited_s += _POLL_S
        start_ns = int((event.start_s - self._before_s) * 1e9)
        frames = buffer.decode_window(start_ns, end_ns)
        for confirmed in self._confirmer_for(event.camera_id).confirm(frames):
            log.info(
                "fall confirmed at full rate",
                extra={"camera": event.camera_id, "candidate": event.event_id},
            )
            self._on_event(self._record(confirmed, candidate))

    def _record(self, confirmed: Event, candidate: EventRecord) -> EventRecord:
        """Times map to UTC through the candidate: same camera, same arrival clock."""

        def utc(t_s: float) -> datetime:
            return candidate.ended_utc + timedelta(seconds=t_s - candidate.event.end_s)

        return EventRecord(
            confirmed,
            utc(confirmed.start_s),
            utc(confirmed.end_s),
            self._clock(),
            candidate.config_hash,
        )
