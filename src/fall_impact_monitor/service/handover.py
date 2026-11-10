"""Hand-over from a camera's reader to its analysis: real time, never a backlog.

A thread reads and decodes the stream as it arrives, so frames are stamped with the time
they were received as long as reading and decoding keep up with the camera. Analysis takes
the most recent decoded frame; when it is slower than the camera, the frames it had no
time for are skipped, never queued. Stream states (connected, stale, disconnected) are
kept in order and are not dropped to make room for frames. Skipped frames are counted and
logged, and never reported as analysed. Reading starts on the first iteration.
"""

import logging
import threading
import time
from collections import deque
from collections.abc import Callable, Iterable, Iterator

from fall_impact_monitor.capture.reader import StateEvent
from fall_impact_monitor.service.camera import DecodedFrame

log = logging.getLogger(__name__)

Item = DecodedFrame | StateEvent
# Items waiting while analysis is busy. Consecutive frames replace each other, so only a
# flapping stream with a stuck analysis fills it (the watchdog's job); frames make room
# first, stream states only when there is no frame left to drop.
_MAX_PENDING_ITEMS = 64
_SKIP_LOG_INTERVAL_S = 60.0
_WAIT_S = 0.5


class LatestFrames:
    def __init__(
        self,
        source: Iterable[Item],
        camera_id: str,
        *,
        on_close: Callable[[], None] | None = None,
    ) -> None:
        """``on_close`` stops the underlying reader (it may be waiting for the network)."""
        self.camera_id = camera_id
        self._on_close = on_close
        self.skipped = 0
        self.started_s = time.monotonic()
        self._items: deque[Item] = deque()
        self._started = False
        self._cond = threading.Condition()
        self._closed = threading.Event()
        self._done = False
        self._error: Exception | None = None
        self._logged_skipped = 0
        self._logged_s = self.started_s
        self._thread = threading.Thread(
            target=self._read, args=(source,), name=f"reader-{camera_id}", daemon=True
        )

    @property
    def reading(self) -> bool:
        return self._thread.is_alive()

    def close(self) -> None:
        """Stop reading after the current item; the iterator then ends."""
        self._closed.set()
        if self._on_close is not None:
            self._on_close()
        with self._cond:
            self._cond.notify_all()

    def join(self, timeout_s: float) -> None:
        if self._started:
            self._thread.join(timeout_s)

    def _read(self, source: Iterable[Item]) -> None:
        try:
            for item in source:
                if self._closed.is_set():
                    break
                with self._cond:
                    if (
                        isinstance(item, DecodedFrame)
                        and self._items
                        and isinstance(self._items[-1], DecodedFrame)
                    ):
                        self._items[-1] = item  # analysis had no time for the older one
                        self.skipped += 1
                    else:
                        if len(self._items) >= _MAX_PENDING_ITEMS:
                            self._make_room()
                        self._items.append(item)
                    self._cond.notify()
                self._log_skipped()
        except Exception as error:  # handed to the analysis thread, which reopens
            self._error = error
        finally:
            with self._cond:
                self._done = True
                self._cond.notify_all()

    def _make_room(self) -> None:
        """Drop the oldest waiting frame, or the oldest stream state if there is none."""
        for index, waiting in enumerate(self._items):
            if isinstance(waiting, DecodedFrame):
                del self._items[index]
                self.skipped += 1
                return
        dropped = self._items.popleft()
        log.warning(
            "stream state dropped while analysis is stuck",
            extra={"camera": self.camera_id, "state": getattr(dropped, "state", "")},
        )

    def _log_skipped(self) -> None:
        now_s = time.monotonic()
        if now_s - self._logged_s < _SKIP_LOG_INTERVAL_S or self.skipped == self._logged_skipped:
            return
        log.info(
            "frames skipped",
            extra={
                "camera": self.camera_id,
                "skipped": self.skipped - self._logged_skipped,
                "window_s": round(now_s - self._logged_s),
            },
        )
        self._logged_skipped, self._logged_s = self.skipped, now_s

    def __iter__(self) -> Iterator[Item]:
        if not self._started:
            self._started = True
            self._thread.start()
        return self._take()

    def _take(self) -> Iterator[Item]:
        while True:
            with self._cond:
                while not self._items and not self._done and not self._closed.is_set():
                    self._cond.wait(_WAIT_S)
                if self._items:
                    item = self._items.popleft()
                elif self._error is not None and not self._closed.is_set():
                    raise self._error
                else:
                    return
            yield item
