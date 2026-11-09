"""One camera: frames from the MediaMTX relay → perception → detectors → event records.

Runs in its own thread (decoding and inference are CPU-bound and blocking); events leave
through ``on_event``, which only queues them (see ``Dispatcher.submit``), so nothing here
waits for an output.
"""

import logging
import threading
import time
from collections import deque
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np
from numpy.typing import NDArray

from fall_impact_monitor.api.perception import FrameResult
from fall_impact_monitor.capture import FrameRecord
from fall_impact_monitor.capture.reader import FrameEvent, ReaderEvent, StateEvent
from fall_impact_monitor.config import CameraSettings
from fall_impact_monitor.events import Event
from fall_impact_monitor.events.record import EventRecord
from fall_impact_monitor.recording.mediamtx_config import substream_path
from fall_impact_monitor.service.timebase import to_utc

log = logging.getLogger(__name__)
_RATE_WINDOW_S = 10.0


@dataclass(frozen=True)
class DecodedFrame:
    record: FrameRecord
    image: NDArray[np.uint8]


class Pipeline(Protocol):
    def process(self, frame_bgr: NDArray[np.uint8], record: FrameRecord) -> FrameResult: ...


class Analyzer(Protocol):
    def update(self, result: FrameResult, frame_size: tuple[int, int]) -> list[Event]: ...


def analysis_url(rtsp_url: str, camera: CameraSettings) -> str:
    """The relayed substream when the camera has one, else the main stream."""
    path = substream_path(camera.id) if camera.sub_url else camera.id
    return f"{rtsp_url.rstrip('/')}/{path}"


def decode(events: Iterable[ReaderEvent]) -> Iterator[DecodedFrame | StateEvent]:
    for event in events:
        if isinstance(event, StateEvent):
            yield event
            continue
        assert isinstance(event, FrameEvent)
        for frame in event.packet.decode():
            image = np.asarray(frame.to_ndarray(format="bgr24"), dtype=np.uint8)
            yield DecodedFrame(event.record, image)


class CameraWorker:
    def __init__(
        self,
        camera_id: str,
        *,
        pipeline: Pipeline,
        analyzer: Analyzer,
        on_event: Callable[[EventRecord], None],
        config_hash: str,
        max_width_px: int,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.camera_id = camera_id
        self._pipeline = pipeline
        self._analyzer = analyzer
        self._on_event = on_event
        self._config_hash = config_hash
        self._max_width_px = max_width_px
        self._monotonic = monotonic
        self._state = "starting"
        self._last_frame_s: float | None = None
        self._analysed: deque[float] = deque()
        self._stop = threading.Event()

    def stop(self) -> None:
        self._stop.set()

    def run(self, source: Iterable[DecodedFrame | StateEvent]) -> None:
        for item in source:
            if self._stop.is_set():
                return
            try:
                self.process(item)
            except Exception:
                log.exception("frame processing failed", extra={"camera": self.camera_id})

    def process(self, item: DecodedFrame | StateEvent) -> None:
        if isinstance(item, StateEvent):
            self._state = item.state
            log.info("stream state", extra={"camera": self.camera_id, "state": item.state})
            return
        image = item.image
        height, width = image.shape[:2]
        if width > self._max_width_px:
            height = round(height * self._max_width_px / width)
            width = self._max_width_px
            image = np.asarray(
                cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA), dtype=np.uint8
            )
        now_s = item.record.arrival_ns / 1e9
        self._last_frame_s = now_s
        result = self._pipeline.process(image, item.record)
        if result.analysed:
            self._analysed.append(now_s)
            while self._analysed and now_s - self._analysed[0] > _RATE_WINDOW_S:
                self._analysed.popleft()
        for event in self._analyzer.update(result, (width, height)):
            self._on_event(
                EventRecord(
                    event=event,
                    started_utc=to_utc(event.start_s, item.record),
                    ended_utc=to_utc(event.end_s, item.record),
                    detected_utc=item.record.wall_utc,
                    config_hash=self._config_hash,
                )
            )

    def status_line(self) -> str:
        line = f"{self.camera_id}: {self._state}"
        if self._last_frame_s is not None:
            age_s = self._monotonic() - self._last_frame_s
            fps = len(self._analysed) / _RATE_WINDOW_S
            line += f", last frame {age_s:.0f} s ago, analysing {fps:.1f} fps"
        return line
