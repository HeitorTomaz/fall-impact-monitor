"""One Euro filter: an adaptive low-pass filter for noisy, irregularly sampled signals.

Written from the paper: G. Casiez, N. Roussel, D. Vogel, "1€ Filter: A Simple
Speed-based Low-pass Filter for Noisy Input in Interactive Systems", CHI 2012 (see
CREDITS.md). Slow movement is smoothed strongly (less jitter); fast movement passes
with little lag. Works on the real time of each sample; a non-positive time step leaves
the state unchanged (never invents time).
"""

import math


def _alpha(dt_s: float, cutoff_hz: float) -> float:
    tau = 1.0 / (2.0 * math.pi * cutoff_hz)
    return 1.0 / (1.0 + tau / dt_s)


class OneEuroFilter:
    def __init__(
        self, *, min_cutoff: float = 1.0, beta: float = 0.0, d_cutoff: float = 1.0
    ) -> None:
        self._min_cutoff = min_cutoff
        self._beta = beta
        self._d_cutoff = d_cutoff
        self._t: float | None = None
        self._x = 0.0
        self._dx = 0.0

    def filter(self, t_s: float, x: float) -> float:
        if self._t is None:
            self._t, self._x, self._dx = t_s, x, 0.0
            return x
        dt = t_s - self._t
        if dt <= 0:
            return self._x
        dx = (x - self._x) / dt
        self._dx = _alpha(dt, self._d_cutoff) * dx + (1 - _alpha(dt, self._d_cutoff)) * self._dx
        cutoff = self._min_cutoff + self._beta * abs(self._dx)
        a = _alpha(dt, cutoff)
        self._x = a * x + (1 - a) * self._x
        self._t = t_s
        return self._x
