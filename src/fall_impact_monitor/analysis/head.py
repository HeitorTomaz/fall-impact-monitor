"""Head kinematics for the experimental head-impact rules.

The head point is the mean of the visible head keypoints (nose, eyes, ears), or the
top-centre of the box when none is visible. Positions are smoothed in pixels with a One
Euro filter and speeds divided by a reference height (the tallest box of the last 2 s), so
the distance to the camera does not matter and a box that shrinks (crouching, sitting,
a box cut by furniture) does not look like head motion. Velocities are
differentiated only between consecutive frames whose timestamps are valid: an invalid
timestamp breaks the chain, so no velocity is ever computed across it.

Evidence ``head_stop``: the head was moving fast (peak speed in the last
``peak_window_s``) and then stops or reverses. Where it happened (inside a safe zone or
not) is decided by the caller. A camera at 5-25 fps sees the motion before and after a
contact, never the contact itself: this is evidence, not a measured impact.
"""

from collections import deque
from dataclasses import dataclass

from fall_impact_monitor.analysis.one_euro import OneEuroFilter
from fall_impact_monitor.perception.tracking import Track

_HEAD_KEYPOINTS = (0, 1, 2, 3, 4)  # COCO-17: nose, eyes, ears
_MIN_KEYPOINT_SCORE = 0.25
_BOX_HEAD_FRACTION = 0.1  # fallback: a tenth of the box below its top


@dataclass(frozen=True)
class HeadConfig:
    min_speed_hps: float = 1.0  # head speed before the stop, person heights per second
    stop_ratio: float = 0.3  # speed after the stop at most this share of the peak
    peak_window_s: float = 0.5
    filter_min_cutoff: float = 1.5
    filter_beta: float = 0.02  # per pixel/s (positions are filtered in pixels)
    reference_window_s: float = 2.0  # reference height: tallest box in this window


@dataclass(frozen=True)
class HeadSample:
    t_s: float
    x: float  # filtered position, pixels
    y: float
    vx_hps: float  # velocity, person heights per second
    vy_hps: float
    peak_speed_hps: float  # highest speed within peak_window_s
    height_px: float

    @property
    def speed_hps(self) -> float:
        return float((self.vx_hps**2 + self.vy_hps**2) ** 0.5)


def head_point(track: Track) -> tuple[float, float] | None:
    if track.keypoints is not None and track.keypoint_scores is not None:
        visible = [
            track.keypoints[i]
            for i in _HEAD_KEYPOINTS
            if track.keypoint_scores[i] >= _MIN_KEYPOINT_SCORE
        ]
        if visible:
            return (
                float(sum(p[0] for p in visible) / len(visible)),
                float(sum(p[1] for p in visible) / len(visible)),
            )
    x1, y1, x2, y2 = track.box_xyxy
    return ((x1 + x2) / 2, y1 + _BOX_HEAD_FRACTION * (y2 - y1))


class HeadTracker:
    """Per track: filtered head position and velocity on valid-timestamp frames."""

    def __init__(self, config: HeadConfig) -> None:
        self._config = config
        self._reset()

    def _reset(self) -> None:
        c = self._config
        self._fx = OneEuroFilter(min_cutoff=c.filter_min_cutoff, beta=c.filter_beta)
        self._fy = OneEuroFilter(min_cutoff=c.filter_min_cutoff, beta=c.filter_beta)
        self._prev: tuple[float, float, float] | None = None  # t, x, y (pixels)
        self._speeds: deque[tuple[float, float]] = deque()
        self._heights: deque[tuple[float, float]] = deque()

    def update(
        self, t_s: float, point: tuple[float, float], height_px: float, *, valid_time: bool
    ) -> HeadSample | None:
        if not valid_time or height_px <= 1.0:
            self._reset()
            return None
        self._heights.append((t_s, height_px))
        while t_s - self._heights[0][0] > self._config.reference_window_s:
            self._heights.popleft()
        reference = max(h for _, h in self._heights)
        x = self._fx.filter(t_s, point[0])
        y = self._fy.filter(t_s, point[1])
        prev, self._prev = self._prev, (t_s, x, y)
        if prev is None or t_s <= prev[0]:
            return None
        dt = t_s - prev[0]
        vx, vy = (x - prev[1]) / dt / reference, (y - prev[2]) / dt / reference
        speed = (vx**2 + vy**2) ** 0.5
        self._speeds.append((t_s, speed))
        while self._speeds and t_s - self._speeds[0][0] > self._config.peak_window_s:
            self._speeds.popleft()
        peak = max(s for _, s in self._speeds)
        return HeadSample(t_s, x, y, vx, vy, peak, reference)


def head_stop(prev: HeadSample, cur: HeadSample, config: HeadConfig) -> float | None:
    """Strength (0-1) of a fast head stop or reversal, else None."""
    peak = cur.peak_speed_hps
    if peak < config.min_speed_hps:
        return None
    stopped = cur.speed_hps <= config.stop_ratio * peak
    reversed_ = prev.vx_hps * cur.vx_hps + prev.vy_hps * cur.vy_hps < 0
    if not (stopped or reversed_):
        return None
    return min(1.0, peak / (2 * config.min_speed_hps))
