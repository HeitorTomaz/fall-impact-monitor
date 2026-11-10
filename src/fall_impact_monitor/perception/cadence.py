"""How often frames are analysed, and whether a camera keeps its minimum rate."""

from collections import deque
from dataclasses import dataclass
from typing import Literal

SceneState = Literal["empty", "person", "attention"]


@dataclass(frozen=True)
class CadencePolicy:
    """Analysis rate per scene state; ``attention`` analyses every frame."""

    empty_fps: float = 1.0
    person_fps: float = 5.0

    def interval_s(self, state: SceneState) -> float:
        if state == "attention":
            return 0.0
        return 1.0 / (self.person_fps if state == "person" else self.empty_fps)


class RateMonitor:
    """Reports ``degraded`` when fewer than ``min_fps`` frames per second were analysed
    over the last ``window_s`` seconds. Never during the first window, nor during the first
    window after ``resume`` (the stream reconnected: the window would otherwise judge the
    outage). A long gap between analysed frames while video flows is slowness, not an
    outage, and is reported."""

    def __init__(self, min_fps: float, window_s: float = 10.0, *, started_s: float) -> None:
        if min_fps <= 0 or window_s <= 0:
            raise ValueError("min_fps and window_s must be positive")
        self._min_fps = min_fps
        self._window_s = window_s
        self._started_s = started_s
        self._times: deque[float] = deque()

    def record(self, t_s: float) -> None:
        self._times.append(t_s)

    def resume(self, now_s: float) -> None:
        """The stream (re)connected: judge analysis on a full window from now."""
        self._started_s = now_s

    def degraded(self, now_s: float) -> bool:
        if now_s - self._started_s < self._window_s:
            return False
        cutoff = now_s - self._window_s
        while self._times and self._times[0] <= cutoff:
            self._times.popleft()
        return len(self._times) / self._window_s < self._min_fps
