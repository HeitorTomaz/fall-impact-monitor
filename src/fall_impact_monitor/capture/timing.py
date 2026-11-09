"""Frame records and timestamp-quality classification.

Timestamps are never modified or invented: the classifier only labels each
packet so that later stages can exclude unreliable deltas from kinematics.
"""

from collections import deque
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from fractions import Fraction
from statistics import fmean


class TimestampQuality(StrEnum):
    """Quality of a frame's source timestamp relative to the previous frame."""

    OK = "ok"
    ZERO_DELTA = "zero_delta"
    BACKWARD = "backward"
    OUTLIER_GAP = "outlier_gap"
    MISSING = "missing"


@dataclass(frozen=True)
class FrameRecord:
    """Timing metadata of one encoded video frame (no pixels)."""

    camera_id: str
    seq: int
    pts: int | None
    time_base: Fraction
    arrival_ns: int
    wall_utc: datetime
    keyframe: bool
    size_bytes: int
    quality: TimestampQuality
    connection: int = 0  # index of the (re)connection that delivered the frame

    @property
    def pts_s(self) -> float | None:
        """Source presentation time in seconds, if the frame has one."""
        return None if self.pts is None else float(self.pts * self.time_base)


class TimestampClassifier:
    """Stateful classifier of consecutive source timestamps.

    The expected frame interval is the mean of the last ``window`` positive
    deltas, where each gap enters the mean clipped at ``outlier_factor`` times
    the interval known at that moment. Including (clipped) gaps keeps the mean
    near the true interval for bursty sources that deliver runs of tiny deltas
    and then catch up; clipping stops one long freeze from hiding later gaps;
    ignoring zero deltas lets the estimate survive frozen timestamps. A positive
    delta above ``outlier_factor`` times that mean is an ``OUTLIER_GAP`` once
    ``warmup`` positive deltas have been seen.
    """

    def __init__(self, outlier_factor: float = 4.0, warmup: int = 10, window: int = 50) -> None:
        self._outlier_factor = outlier_factor
        self._warmup = warmup
        self._deltas_s: deque[float] = deque(maxlen=window)
        self._previous_s: float | None = None

    def classify(self, pts: int | None, time_base: Fraction) -> TimestampQuality:
        if pts is None:
            return TimestampQuality.MISSING
        current_s = float(pts * time_base)
        previous_s = self._previous_s
        self._previous_s = current_s
        if previous_s is None:
            return TimestampQuality.OK
        delta_s = current_s - previous_s
        if delta_s < 0:
            return TimestampQuality.BACKWARD
        if delta_s == 0:
            return TimestampQuality.ZERO_DELTA
        expected_s = fmean(self._deltas_s) if len(self._deltas_s) >= self._warmup else None
        if expected_s is None:
            self._deltas_s.append(delta_s)
            return TimestampQuality.OK
        cap_s = self._outlier_factor * expected_s
        self._deltas_s.append(min(delta_s, cap_s))
        return TimestampQuality.OUTLIER_GAP if delta_s > cap_s else TimestampQuality.OK
