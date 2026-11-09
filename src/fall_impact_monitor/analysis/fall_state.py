# Copyright 2026 fall-detection contributors (edgefallkit)
# Copyright 2026 The Fall Impact Monitor contributors
# SPDX-License-Identifier: Apache-2.0
#
# Adapted from https://github.com/suharvest/edgefallkit
#   platforms/rknn/fall_core.py (FallConfig, FallDetector) — see THIRD_PARTY_NOTICES.md.
# Changed: the event fires on the descent itself (a person who gets up at once still
# raises it) instead of after sustained lying; hip motion is normalised by the person's
# upright height instead of the image height; the temporal MLP, occlusion latch and
# IoU tracker were removed; a suspicion that ends without an event is reported as a
# near-miss; typed results.
"""Per-person fall state machine: normal -> suspected -> fallen -> recovering -> normal."""

from dataclasses import dataclass, field
from typing import Any, Literal

from fall_impact_monitor.analysis.observation import Observation

State = Literal["normal", "suspected", "fallen", "recovering"]


@dataclass(frozen=True)
class FallConfig:
    drop_speed_hps: float = 0.6  # hip drop speed, person heights per second
    motion_window_s: float = 0.75  # a fast drop arms suspicion within this window
    horizontal_angle_deg: float = 55.0  # torso angle counted as horizontal
    horizontal_aspect: float = 1.25  # box width/height counted as horizontal
    fall_drop_heights: float = 0.35  # hip drop that confirms a fall on the descent
    lying_confirm_s: float = 0.8  # or: lying this long after a fast drop
    suspected_timeout_s: float = 2.0
    upright_angle_deg: float = 35.0
    upright_aspect: float = 1.1
    recovery_window_s: float = 2.0
    cooldown_s: float = 5.0


@dataclass(frozen=True)
class StepResult:
    state: State
    event: bool = False
    near_miss: bool = False
    features: dict[str, Any] = field(default_factory=dict)


class FallStateMachine:
    """Only geometry arms ``suspected``: a horizontal cue plus a fast hip drop. The
    event fires once the hip has dropped ``fall_drop_heights`` person heights while
    horizontal, or after lying ``lying_confirm_s``. Recovery never cancels an event."""

    def __init__(self, config: FallConfig) -> None:
        self.config = config
        self.state: State = "normal"
        self._prev: Observation | None = None
        self._baseline_hip: float | None = None
        self._baseline_height: float | None = None
        self._last_fast_drop_s = -1.0
        self._suspected_since_s = -1.0
        self._lying_since_s = -1.0
        self._recovery_since_s = -1.0
        self._cooldown_until_s = -1.0
        self._reset_features()

    def _reset_features(self) -> None:
        self._max_drop = 0.0
        self._max_speed = 0.0
        self._max_angle = 0.0
        self._max_aspect = 0.0

    def _features(self, now_s: float) -> dict[str, Any]:
        lying_s = now_s - self._lying_since_s if self._lying_since_s >= 0 else 0.0
        return {
            "max_drop_heights": round(self._max_drop, 4),
            "max_drop_speed_hps": round(self._max_speed, 4),
            "max_torso_angle_deg": round(self._max_angle, 2),
            "max_aspect": round(self._max_aspect, 3),
            "lying_s": round(lying_s, 3),
            "recovered": self.state in ("recovering", "normal"),
        }

    def update(self, o: Observation) -> StepResult:
        c = self.config
        if not o.valid:
            # Missing observations can hold a state but never start a fall.
            return StepResult(self.state)
        horizontal = o.torso_angle_deg >= c.horizontal_angle_deg or o.aspect >= c.horizontal_aspect
        lying = o.torso_angle_deg >= c.horizontal_angle_deg and o.aspect >= c.horizontal_aspect
        upright = o.torso_angle_deg <= c.upright_angle_deg and o.aspect <= c.upright_aspect

        if self.state == "normal" and not horizontal:
            self._baseline_hip, self._baseline_height = o.hip_y, o.height_px
        speed = 0.0
        if self._prev is not None and self._baseline_height:
            dt = o.t_s - self._prev.t_s
            if dt > 1e-4:
                speed = (o.hip_y - self._prev.hip_y) / dt / self._baseline_height
        if speed >= c.drop_speed_hps:
            self._last_fast_drop_s = o.t_s
        self._prev = o
        drop = (
            (o.hip_y - self._baseline_hip) / self._baseline_height
            if self._baseline_hip is not None and self._baseline_height
            else 0.0
        )
        self._lying_since_s = (
            (self._lying_since_s if self._lying_since_s >= 0 else o.t_s) if lying else -1.0
        )

        event = False
        if self.state == "normal":
            fast = (
                self._last_fast_drop_s >= 0 and o.t_s - self._last_fast_drop_s <= c.motion_window_s
            )
            if horizontal and fast and o.t_s >= self._cooldown_until_s:
                self.state = "suspected"
                self._suspected_since_s = self._last_fast_drop_s
                self._reset_features()
        if self.state == "suspected":
            self._max_drop = max(self._max_drop, drop)
            self._max_speed = max(self._max_speed, speed)
            self._max_angle = max(self._max_angle, o.torso_angle_deg)
            self._max_aspect = max(self._max_aspect, o.aspect)
            lying_long = (
                self._lying_since_s >= 0 and o.t_s - self._lying_since_s >= c.lying_confirm_s
            )
            if (horizontal and self._max_drop >= c.fall_drop_heights) or lying_long:
                self.state = "fallen"
                event = True
                self._cooldown_until_s = o.t_s + c.cooldown_s
            elif upright or o.t_s - self._suspected_since_s > c.suspected_timeout_s:
                features = self._features(o.t_s)
                self.state = "normal"
                return StepResult(self.state, near_miss=True, features=features)
        elif self.state == "fallen" and upright:
            self.state = "recovering"
            self._recovery_since_s = o.t_s
        elif self.state == "recovering":
            if not upright:
                self.state = "fallen"
            elif o.t_s - self._recovery_since_s >= c.recovery_window_s:
                self.state = "normal"
        return StepResult(self.state, event=event, features=self._features(o.t_s))
