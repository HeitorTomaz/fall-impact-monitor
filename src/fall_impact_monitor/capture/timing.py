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

    @property
    def pts_s(self) -> float | None:
        """Source presentation time in seconds, if the frame has one."""
        return None if self.pts is None else float(self.pts * self.time_base)


class TimestampClassifier:
    """Stateful classifier of consecutive source timestamps.

    The expected frame interval is the mean of the last ``window`` non-negative
    deltas, including zero deltas and gaps: bursty sources hold timestamps and
    then catch up, so only the mean of every delta tracks the true interval
    (a median, or a mean of "good" deltas only, drifts low and cascades). A
    positive delta above ``outlier_factor`` times that mean is an
    ``OUTLIER_GAP`` once ``warmup`` deltas have been seen.
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
        expected_s = fmean(self._deltas_s) if len(self._deltas_s) >= self._warmup else None
        self._deltas_s.append(delta_s)
        if delta_s == 0:
            return TimestampQuality.ZERO_DELTA
        if expected_s is not None and delta_s > self._outlier_factor * expected_s:
            return TimestampQuality.OUTLIER_GAP
        return TimestampQuality.OK
