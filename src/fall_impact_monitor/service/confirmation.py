"""Full-rate re-analysis of fall candidates (see ``api/confirmation.py``), in two threads.

- **Collector.** Camera workers hand near-miss records over without ever waiting (a
  full queue refuses the request; the near-miss itself is already recorded). The
  collector waits until the packet buffer holds the frames ``after_s`` past the
  candidate (or a grace period more, for a camera that stopped) and copies the window's
  encoded packets at once, so that a candidate queued behind a long analysis does not
  lose the start of its window when the buffer moves on.
- **Analyser.** Decodes each copied window and analyses every frame. A fall seen there
  is sent like any other event, unless the live analysis already alerted for the same
  fall (its interval overlaps): the re-analysis only ever adds alerts.
"""

import contextlib
import logging
import queue
import threading
import time
from collections import deque
from collections.abc import Callable, Iterable, Mapping
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol

from fall_impact_monitor.api.confirmation import is_candidate
from fall_impact_monitor.events import Event, EventCategory
from fall_impact_monitor.events.record import EventRecord

log = logging.getLogger(__name__)
_GRACE_S = 5.0  # beyond after_s, for a camera that stopped streaming
_POLL_S = 0.5
_LIVE_ALERTS_KEPT = 20  # per camera, to recognise a fall the live analysis raised


class _Window(Protocol):
    def decode(self) -> Iterable[Any]: ...


class _Buffer(Protocol):
    def newest_arrival_ns(self) -> int | None: ...

    def snapshot(self, start_ns: int, end_ns: int) -> Any: ...


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
        self._requests: queue.Queue[EventRecord | None] = queue.Queue(maxsize=max_queue)
        self._windows: queue.Queue[tuple[EventRecord, _Window] | None] = queue.Queue(
            maxsize=max_queue
        )
        self._live: dict[str, deque[tuple[float, float]]] = {}
        self._live_lock = threading.Lock()
        self._threads: list[threading.Thread] = []

    def request(self, record: EventRecord) -> bool:
        """Queue a candidate; never blocks. False when it is not one or the queue is full."""
        event = record.event
        if not is_candidate(event) or event.camera_id not in self._buffers:
            return False
        try:
            self._requests.put_nowait(record)
        except queue.Full:
            log.warning(
                "re-analysis queue full, candidate kept as a near-miss",
                extra={"event": event.event_id},
            )
            return False
        return True

    def note_alert(self, record: EventRecord) -> None:
        """A possible fall the live analysis notified (same camera clock as candidates)."""
        event = record.event
        if event.category is not EventCategory.POSSIBLE_FALL or event.near_miss:
            return
        with self._live_lock:
            kept = self._live.setdefault(event.camera_id, deque(maxlen=_LIVE_ALERTS_KEPT))
            kept.append((event.start_s, event.end_s))

    def start(self) -> None:
        self._threads = [
            threading.Thread(
                target=self._loop, args=(self.collect_next,), name="reanalysis-collect", daemon=True
            ),
            threading.Thread(
                target=self._loop, args=(self.analyse_next,), name="reanalysis", daemon=True
            ),
        ]
        for thread in self._threads:
            thread.start()

    def stop(self) -> None:
        for stop_queue in (self._requests, self._windows):
            with contextlib.suppress(queue.Full):  # full: the daemon threads end with us
                stop_queue.put_nowait(None)
        for thread in self._threads:
            thread.join(timeout=5)

    @staticmethod
    def _loop(step: Callable[[float | None], bool]) -> None:
        while step(None):
            pass

    def process_next(self, timeout_s: float | None) -> bool:
        """Both stages for one candidate (tests and single-threaded use)."""
        return self.collect_next(timeout_s) and self.analyse_next(timeout_s)

    @staticmethod
    def _take(source: "queue.Queue[Any]", timeout_s: float | None) -> Any:
        """Blocks when ``timeout_s`` is None; raises ``queue.Empty`` otherwise."""
        if timeout_s is None:
            return source.get()
        return source.get(timeout=timeout_s) if timeout_s > 0 else source.get_nowait()

    def collect_next(self, timeout_s: float | None) -> bool:
        """Copy the packets of one candidate's window; False when stopped or idle."""
        try:
            record = self._take(self._requests, timeout_s)
        except queue.Empty:
            return False
        if record is None:
            return False
        try:
            window = self._collect(record)
            if window is not None:
                self._windows.put((record, window), timeout=_GRACE_S)
        except queue.Full:
            log.warning(
                "re-analysis busy, candidate kept as a near-miss",
                extra={"event": record.event.event_id},
            )
        except Exception:
            log.exception("re-analysis failed", extra={"event": record.event.event_id})
        return True

    def analyse_next(self, timeout_s: float | None) -> bool:
        """Analyse one copied window; False when stopped or idle."""
        try:
            item = self._take(self._windows, timeout_s)
        except queue.Empty:
            return False
        if item is None:
            return False
        candidate, window = item
        try:
            self._analyse(candidate, window)
        except Exception:
            log.exception("re-analysis failed", extra={"event": candidate.event.event_id})
        return True

    def _collect(self, candidate: EventRecord) -> _Window | None:
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
        window: _Window | None = buffer.snapshot(start_ns, end_ns)
        return window

    def _analyse(self, candidate: EventRecord, window: _Window) -> None:
        event = candidate.event
        for seen in self._confirmer_for(event.camera_id).confirm(window.decode()):
            if self._already_alerted(seen):
                log.info(
                    "possible fall seen at full rate, already alerted live",
                    extra={"camera": event.camera_id, "candidate": event.event_id},
                )
                continue
            log.info(
                "possible fall seen at full rate",
                extra={"camera": event.camera_id, "candidate": event.event_id},
            )
            self._on_event(self._record(seen, candidate))

    def _already_alerted(self, seen: Event) -> bool:
        margin = self._after_s
        with self._live_lock:
            alerts = list(self._live.get(seen.camera_id, ()))
        return any(
            seen.start_s <= end_s + margin and seen.end_s >= start_s - margin
            for start_s, end_s in alerts
        )

    def _record(self, seen: Event, candidate: EventRecord) -> EventRecord:
        """Times map to UTC through the candidate: same camera, same arrival clock."""

        def utc(t_s: float) -> datetime:
            return candidate.ended_utc + timedelta(seconds=t_s - candidate.event.end_s)

        return EventRecord(
            seen, utc(seen.start_s), utc(seen.end_s), self._clock(), candidate.config_hash
        )
