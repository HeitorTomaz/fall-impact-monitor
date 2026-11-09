"""A cheap thump detector for 8 kHz mono audio (experimental head-impact evidence).

Energy onsets: the level of each 32 ms frame (16 ms hop) is compared with the background,
the median level of the last second; a frame ``min_db`` or more above it is a thump.
Steady sounds (speech, TV, fans) raise the background and do not trigger it; two bursts
within ``refractory_s`` count once. The idea follows onset-strength detection as
documented by librosa (see CREDITS.md); the code is written for this project. This is a
filter, not a classifier: a door or a dropped object is a thump too, which is why audio
alone never raises an event.
"""

from collections import deque
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

_FRAME_S = 0.032
_HOP_S = 0.016
_BACKGROUND_S = 1.0
_MIN_HISTORY_FRAMES = 10


@dataclass(frozen=True)
class Thump:
    t_s: float  # monotonic seconds of the frame start
    strength: float  # 0-1


class ThumpDetector:
    def __init__(
        self, *, sample_rate: int = 8000, min_db: float = 12.0, refractory_s: float = 0.3
    ) -> None:
        self._rate = sample_rate
        self._min_db = min_db
        self._refractory_s = refractory_s
        self._frame = int(_FRAME_S * sample_rate)
        self._hop = int(_HOP_S * sample_rate)
        self._buffer = np.zeros(0, np.float32)
        self._buffer_t_s = 0.0
        self._history: deque[float] = deque(maxlen=int(_BACKGROUND_S / _HOP_S))
        self._last_thump_s = float("-inf")

    def feed(self, t0_s: float, samples: NDArray[np.float32]) -> list[Thump]:
        """``samples`` start at ``t0_s``; consecutive calls continue the same stream."""
        # Re-anchor on every chunk: the leftover samples end where this chunk starts. This
        # keeps times right across reconnections and audio-clock drift.
        self._buffer_t_s = t0_s - self._buffer.size / self._rate
        self._buffer = np.concatenate([self._buffer, samples.astype(np.float32, copy=False)])
        found: list[Thump] = []
        while self._buffer.size >= self._frame:
            frame = self._buffer[: self._frame]
            level_db = 10.0 * float(np.log10(np.mean(frame.astype(np.float64) ** 2) + 1e-12))
            if len(self._history) >= _MIN_HISTORY_FRAMES:
                rise_db = level_db - float(np.median(self._history))
                t_s = self._buffer_t_s
                if rise_db >= self._min_db and t_s - self._last_thump_s >= self._refractory_s:
                    self._last_thump_s = t_s
                    found.append(Thump(t_s, min(1.0, rise_db / (2 * self._min_db))))
            self._history.append(level_db)
            self._buffer = self._buffer[self._hop :]
            self._buffer_t_s += self._hop / self._rate
        return found
