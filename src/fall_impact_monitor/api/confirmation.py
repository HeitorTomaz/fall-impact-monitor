"""Full-rate confirmation of fall candidates (ARCHITECTURE §9.1).

Live analysis sees a few frames per second; a fall it sees only partly ends as a
near-miss. A near-miss ``possible_fall`` outside safe zones is a candidate: the frames
around it are analysed again, every one of them (or up to ``max_fps``), with a fresh
tracker and fall rule. A fall raised there is an alert, marked as seen at full rate
(``reanalysed_at_full_rate``). It only adds alerts: the near-miss is always kept.

The service feeds it frames decoded from the packet buffer; ``fim analyze`` feeds it
frames decoded again from the file.
"""

import dataclasses
from collections.abc import Callable, Iterable, Iterator
from typing import Any, Protocol

import cv2
import numpy as np
from numpy.typing import NDArray

from fall_impact_monitor.api.perception import FrameResult
from fall_impact_monitor.capture import FrameRecord
from fall_impact_monitor.capture.rotation import rotate
from fall_impact_monitor.events import Event, EventCategory

_EPSILON_S = 1e-6


class _Pipeline(Protocol):
    def process(self, frame_bgr: NDArray[np.uint8], record: FrameRecord) -> FrameResult: ...


class _FallRule(Protocol):
    def update(self, result: FrameResult, frame_size: tuple[int, int]) -> list[Event]: ...


def is_candidate(event: Event) -> bool:
    """A fall near-miss outside safe zones (lying down in bed is not a candidate)."""
    return (
        event.category is EventCategory.POSSIBLE_FALL
        and event.near_miss
        and "safe_zone" not in event.evidence
    )


def _bgr(frame: Any, rotate_deg: int, max_width_px: int) -> NDArray[np.uint8]:
    image = (
        frame
        if isinstance(frame, np.ndarray)
        else np.asarray(frame.to_ndarray(format="bgr24"), dtype=np.uint8)
    )
    image = rotate(image, rotate_deg)
    height, width = image.shape[:2]
    if width > max_width_px:
        size = (max_width_px, round(height * max_width_px / width))
        image = np.asarray(cv2.resize(image, size, interpolation=cv2.INTER_AREA), np.uint8)
    return image


class FallConfirmer:
    def __init__(
        self,
        make_pipeline: Callable[[], _Pipeline],
        make_rule: Callable[[], _FallRule],
        *,
        max_width_px: int,
        rotate_deg: int = 0,
        max_fps: float = 0.0,
    ) -> None:
        self._make_pipeline = make_pipeline
        self._make_rule = make_rule
        self._max_width_px = max_width_px
        self._rotate_deg = rotate_deg
        self._min_interval_s = 1.0 / max_fps if max_fps > 0 else 0.0

    def confirm(self, frames: Iterable[tuple[FrameRecord, Any]]) -> list[Event]:
        """Alerts raised by analysing ``frames`` (record, decoded frame) in order."""
        return self.decide(self.perceive(frames))

    def perceive(
        self, frames: Iterable[tuple[FrameRecord, Any]]
    ) -> Iterator[tuple[FrameResult, tuple[int, int]]]:
        """Perception of the window's frames (up to ``max_fps``), with frame sizes."""
        pipeline = self._make_pipeline()
        last_s: float | None = None
        for record, frame in frames:
            now_s = record.arrival_ns / 1e9
            if last_s is not None and now_s - last_s < self._min_interval_s - _EPSILON_S:
                continue
            last_s = now_s
            image = _bgr(frame, self._rotate_deg, self._max_width_px)
            yield pipeline.process(image, record), (image.shape[1], image.shape[0])

    def decide(self, results: Iterable[tuple[FrameResult, tuple[int, int]]]) -> list[Event]:
        """The fall rule on perceived frames (also replayed from saved ones)."""
        return decide(self._make_rule, results)


def decide(
    make_rule: Callable[[], _FallRule], results: Iterable[tuple[FrameResult, tuple[int, int]]]
) -> list[Event]:
    """Alerts of a fresh fall rule over a window's perceived frames: at most one, marked
    as seen at full rate."""
    rule = make_rule()
    alerts: list[Event] = []
    for result, size in results:
        for event in rule.update(result, size):
            if event.category is EventCategory.POSSIBLE_FALL and not event.near_miss:
                evidence = {**event.evidence, "reanalysed_at_full_rate": True}
                alerts.append(dataclasses.replace(event, evidence=evidence))
    return alerts[:1]  # one candidate, one alert


class PendingConfirmations:
    """Offline (``fim analyze``): candidates wait until the frames ``after_s`` past
    their end have been read."""

    def __init__(self, after_s: float) -> None:
        self._after_s = after_s
        self._waiting: list[Event] = []

    def add(self, event: Event) -> bool:
        if not is_candidate(event):
            return False
        self._waiting.append(event)
        return True

    def due(self, now_s: float) -> list[Event]:
        ready = [e for e in self._waiting if now_s >= e.end_s + self._after_s]
        self._waiting = [e for e in self._waiting if e not in ready]
        return ready

    def remaining(self) -> list[Event]:
        """At the end of a file: everything left."""
        left, self._waiting = self._waiting, []
        return left
