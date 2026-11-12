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
from functools import cached_property
from typing import Any, Protocol

import cv2
import numpy as np
from numpy.typing import NDArray

from fall_impact_monitor.analysis.orientation import OrientationCheck
from fall_impact_monitor.api.perception import FrameResult
from fall_impact_monitor.capture import FrameRecord
from fall_impact_monitor.capture.lag import ReaderLag
from fall_impact_monitor.capture.reader import FrameEvent, ReaderEvent, StateEvent
from fall_impact_monitor.capture.rotation import rotate
from fall_impact_monitor.config import CameraSettings
from fall_impact_monitor.events import Event
from fall_impact_monitor.events.record import EventRecord
from fall_impact_monitor.health.observe import CameraProbe
from fall_impact_monitor.recording.mediamtx_config import substream_path
from fall_impact_monitor.service.timebase import to_utc

log = logging.getLogger(__name__)
_RATE_WINDOW_S = 10.0


@dataclass(frozen=True)
class DecodedFrame:
    """A decoded frame. The BGR conversion (and rotation) happens on first use of
    ``image``: frames that analysis skips are never converted (#42)."""

    record: FrameRecord
    source: Any  # an av.VideoFrame, or an already converted BGR image
    rotate_deg: int = 0

    @cached_property
    def image(self) -> NDArray[np.uint8]:
        if isinstance(self.source, np.ndarray):
            image = self.source
        else:
            image = np.asarray(self.source.to_ndarray(format="bgr24"), dtype=np.uint8)
        return rotate(image, self.rotate_deg)


class Pipeline(Protocol):
    def process(self, frame_bgr: NDArray[np.uint8], record: FrameRecord) -> FrameResult: ...


class Analyzer(Protocol):
    def update(self, result: FrameResult, frame_size: tuple[int, int]) -> list[Event]: ...


def analysis_url(rtsp_url: str, camera: CameraSettings) -> str:
    """The relayed substream when the camera has one, else the main stream."""
    path = substream_path(camera.id) if camera.sub_url else camera.id
    return f"{rtsp_url.rstrip('/')}/{path}"


def decode(
    events: Iterable[ReaderEvent], *, rotate_deg: int = 0
) -> Iterator[DecodedFrame | StateEvent]:
    for event in events:
        if isinstance(event, StateEvent):
            yield event
            continue
        assert isinstance(event, FrameEvent)
        try:
            frames = list(event.packet.decode())
        except Exception as error:  # a corrupt packet: skip it, keep the stream
            log.warning(
                "packet not decodable",
                extra={"camera": event.record.camera_id, "error": type(error).__name__},
            )
            continue
        for frame in frames:
            yield DecodedFrame(event.record, frame, rotate_deg)


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
        reconnect_window_s: float = 600.0,
        rotate_deg: int = 0,
    ) -> None:
        self.camera_id = camera_id
        self._rotate_deg = rotate_deg
        self._orientation = OrientationCheck()
        self._lag = ReaderLag()
        self.reader_lag_s = 0.0
        self._reconnect_window_s = reconnect_window_s
        self._connections: deque[float] = deque()
        self._busy = False
        self.failures = 0  # consecutive frames that failed
        self._pipeline = pipeline
        self._analyzer = analyzer
        self._on_event = on_event
        self._config_hash = config_hash
        self._max_width_px = max_width_px
        self._monotonic = monotonic
        self._state = "starting"
        self._last_frame_s: float | None = None
        self.started_s = monotonic()
        self.last_activity_s = self.started_s  # any frame or stream state (watchdog)
        self._analysed: deque[float] = deque()
        self._stop = threading.Event()

    def stop(self) -> None:
        self._stop.set()

    def run_forever(
        self,
        make_source: Callable[[], Iterable[DecodedFrame | StateEvent]],
        *,
        retry_s: float = 5.0,
    ) -> None:
        """Run until stopped, reopening the source whenever it fails or ends, so one
        error never ends the monitoring of a camera."""
        while not self._stop.is_set():
            try:
                self.run(make_source())
            except Exception:
                log.exception("camera source failed, reopening", extra={"camera": self.camera_id})
            self._state = "reopening"
            self._stop.wait(retry_s)

    def run(self, source: Iterable[DecodedFrame | StateEvent]) -> None:
        try:
            for item in source:
                if self._stop.is_set():
                    return
                try:
                    self.process(item)
                    self.failures = 0
                except Exception:
                    self.failures += 1
                    log.exception("frame processing failed", extra={"camera": self.camera_id})
        finally:
            close = getattr(source, "close", None)
            if callable(close):
                close()  # stops a reader thread that would otherwise keep reading

    @property
    def last_frame_s(self) -> float | None:
        return self._last_frame_s

    def probe(self, now_s: float) -> CameraProbe:
        degraded = getattr(self._pipeline, "degraded", None)
        recent = [t for t in list(self._connections) if now_s - t <= self._reconnect_window_s]
        return CameraProbe(
            self.camera_id,
            last_frame_s=self._last_frame_s,
            started_s=self.started_s,
            rate_degraded=bool(degraded(now_s)) if callable(degraded) else False,
            busy=self._busy,
            reconnects=max(0, len(recent) - (1 if len(recent) == len(self._connections) else 0)),
            failures=self.failures,
            reader_lag_s=self.reader_lag_s,
            image_inverted=self._orientation.inverted,
        )

    def process(self, item: DecodedFrame | StateEvent) -> None:
        self.last_activity_s = self._monotonic()
        if isinstance(item, StateEvent):
            self._state = item.state
            if item.state == "connected":
                self._lag.reset()  # a new stream: a new reference for the reader lag
                self.reader_lag_s = 0.0
                resume = getattr(self._pipeline, "resume", None)
                if callable(resume):
                    resume(self._monotonic())  # do not judge analysis on the outage
                self._connections.append(self._monotonic())
                while len(self._connections) > 100:
                    self._connections.popleft()
            log.info("stream state", extra={"camera": self.camera_id, "state": item.state})
            return
        self.reader_lag_s = self._lag.update(item.record.pts_s, item.record.arrival_ns / 1e9)
        self._busy = True  # frames wait while this one is analysed: not a lost camera
        try:
            self._process_frame(item)
        finally:
            self._busy = False

    def _process_frame(self, item: DecodedFrame) -> None:
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
        if self._orientation.update(result.tracks):
            log.warning(
                "image looks upside down",
                extra={"camera": self.camera_id, "rotate_deg": self._rotate_deg},
            )
        if result.analysed:
            self._analysed.append(now_s)
            while self._analysed and now_s - self._analysed[0] > _RATE_WINDOW_S:
                self._analysed.popleft()
        attention_until = getattr(self._analyzer, "attention_until", None)
        request_attention = getattr(self._pipeline, "request_attention", None)
        if callable(attention_until) and callable(request_attention):
            until = attention_until()
            if until is not None:
                request_attention(until)  # e.g. a head moving fast: every frame
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
        if self._orientation.inverted:
            variable = f"FIM_CAMERA_{self.camera_id.upper()}_ROTATE_DEG"
            line += f"; image looks upside down: check {variable} (now {self._rotate_deg})"
        return line
