"""How far a camera reader is behind the camera (#42).

Arrival times are real only while reading and decoding keep up. For each frame, the
offset between arrival (host clock) and presentation time (camera clock) is constant
while the reader keeps up, apart from the camera's own jitter; it grows when the reader
falls behind. The lag is the offset minus a reference: the smallest offset seen, which
may only rise slowly (``max_drift``, far above real camera-host clock drift) so that a
reader that fell behind and stays behind keeps showing its lag for hours.

Frames whose presentation time does not advance (cameras that send runs of equal
timestamps, then a jump) carry no new information and are skipped; a clock that goes
back, or a new stream, starts a new reference.
"""

_MAX_DRIFT = 1e-3  # seconds per second; camera and host clocks drift ~5e-5


class ReaderLag:
    def __init__(self, max_drift: float = _MAX_DRIFT) -> None:
        if max_drift < 0:
            raise ValueError("max_drift must not be negative")
        self._max_drift = max_drift
        self._reference_s: float | None = None
        self._last_pts_s: float | None = None
        self._last_arrival_s = 0.0
        self._lag_s = 0.0

    def reset(self) -> None:
        """Forget the reference (a new stream after a reconnection)."""
        self._reference_s = None
        self._last_pts_s = None
        self._lag_s = 0.0

    def update(self, pts_s: float | None, arrival_s: float) -> float:
        """Seconds the reader is behind, after a frame read at ``arrival_s``."""
        if pts_s is None:
            return 0.0
        if self._last_pts_s is not None:
            if pts_s < self._last_pts_s:
                self.reset()  # the camera restarted its clock
            elif pts_s == self._last_pts_s:
                return self._lag_s  # a run of equal timestamps: nothing new
        offset = arrival_s - pts_s
        if self._reference_s is None:
            self._reference_s = offset
        else:
            allowed = self._reference_s + self._max_drift * (arrival_s - self._last_arrival_s)
            self._reference_s = min(allowed, offset)
        self._last_pts_s, self._last_arrival_s = pts_s, arrival_s
        self._lag_s = max(0.0, offset - self._reference_s)
        return self._lag_s
