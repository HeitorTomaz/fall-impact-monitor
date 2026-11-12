"""How far a camera reader is behind the camera (#42).

Arrival times are real only while reading and decoding keep up. For each frame, the
offset between arrival (host clock) and presentation time (camera clock) is constant
while the reader keeps up, apart from the camera's own jitter; it grows when the reader
falls behind. The lag is the offset minus the smallest one of the last ``window_s``
seconds. A long window tolerates clock drift between camera and host (tens of
milliseconds per ten minutes) while a reader that keeps falling behind keeps a large lag.
"""

from collections import deque

_DEFAULT_WINDOW_S = 600.0
_STALLED_CLOCK_S = 1.0  # camera clock not advancing this long: no information


class ReaderLag:
    def __init__(self, window_s: float = _DEFAULT_WINDOW_S) -> None:
        if window_s <= 0:
            raise ValueError("window_s must be positive")
        self._window_s = window_s
        self._minima: deque[tuple[float, float]] = deque()  # (arrival_s, offset_s), rising
        self._last_pts_s: float | None = None
        self._advanced_at_s = 0.0  # arrival of the last frame whose clock advanced

    def reset(self) -> None:
        """Forget the reference (a new stream after a reconnection)."""
        self._minima.clear()
        self._last_pts_s = None

    def update(self, pts_s: float | None, arrival_s: float) -> float:
        """Seconds the reader is behind, for a frame read at ``arrival_s``; 0 without
        a presentation time, and a new reference when the camera clock goes back or
        stands still."""
        if pts_s is None:
            return 0.0
        if self._last_pts_s is not None and pts_s < self._last_pts_s:
            self.reset()
        if self._last_pts_s is None or pts_s > self._last_pts_s:
            self._advanced_at_s = arrival_s
        elif arrival_s - self._advanced_at_s > _STALLED_CLOCK_S:
            self.reset()  # broken timestamps: start again when the clock moves
            self._last_pts_s = pts_s
            self._advanced_at_s = arrival_s
            return 0.0
        self._last_pts_s = pts_s
        offset = arrival_s - pts_s
        while self._minima and self._minima[-1][1] >= offset:
            self._minima.pop()
        self._minima.append((arrival_s, offset))
        while arrival_s - self._minima[0][0] > self._window_s:
            self._minima.popleft()
        return max(0.0, offset - self._minima[0][1])
