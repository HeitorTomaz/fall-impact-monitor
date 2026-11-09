"""Event clip export in a background thread, off the alert path.

Only notified detections get a clip (not near-misses, not monitoring messages). The
queue is bounded: when it is full a request is refused and logged, never blocking the
caller. An export failure (for example nothing recorded) is logged and the worker goes on.
"""

import logging
import queue
import threading
from collections.abc import Callable
from datetime import datetime

from fall_impact_monitor.events import EventCategory
from fall_impact_monitor.events.record import EventRecord

log = logging.getLogger(__name__)
_DETECTIONS = {
    EventCategory.POSSIBLE_FALL,
    EventCategory.PERSON_DOWN,
    EventCategory.POSSIBLE_HEAD_IMPACT,
}

Export = Callable[[str, str, datetime, datetime], object]


class ClipWorker:
    def __init__(self, export: Export, *, max_queue: int = 100) -> None:
        self._export = export
        self._queue: queue.Queue[EventRecord | None] = queue.Queue(maxsize=max_queue)
        self._thread: threading.Thread | None = None

    def request(self, record: EventRecord) -> bool:
        event = record.event
        if event.near_miss or event.category not in _DETECTIONS:
            return False
        try:
            self._queue.put_nowait(record)
        except queue.Full:
            log.warning("clip queue full, clip not exported", extra={"event": event.event_id})
            return False
        return True

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="clips", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._thread is not None:
            self._queue.put(None)
            self._thread.join(timeout=5)

    def _run(self) -> None:
        while True:
            record = self._queue.get()
            if record is None:
                return
            event = record.event
            try:
                self._export(event.camera_id, event.event_id, record.started_utc, record.ended_utc)
            except Exception as error:
                log.warning(
                    "clip export failed",
                    extra={"event": event.event_id, "error": type(error).__name__},
                )
