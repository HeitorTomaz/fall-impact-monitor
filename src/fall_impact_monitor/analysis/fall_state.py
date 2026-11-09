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
# near-miss; the baseline is the highest hip of the last non-horizontal frames; long
# frame intervals and lost poses are handled; typed results.
"""Per-person fall state machine: normal -> suspected -> fallen -> recovering -> normal."""

from collections import deque
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
    fall_drop_heights: float = 0.35  # hip drop that raises the event on the descent
    lying_confirm_s: float = 0.8  # or: lying this long after a fast drop
    suspected_timeout_s: float = 2.0
    upright_angle_deg: float = 35.0
    upright_aspect: float = 1.1
    recovery_window_s: float = 2.0
    cooldown_s: float = 5.0
    baseline_window_s: float = 1.5  # baseline = highest hip of non-horizontal frames in it
    transition_s: float = 1.5  # upright to lying this fast also arms suspicion (0 = off)
    transition_min_drop_heights: float = 0.15  # hip drop both rules below also need
    collapse_ratio: float = 0.4  # box height shrinking this much into lying...
    collapse_window_s: float = 1.0  # ...within this time also arms suspicion (0 = off)


@dataclass(frozen=True)
class StepResult:
    state: State
    event: bool = False
    near_miss: bool = False
    features: dict[str, Any] = field(default_factory=dict)


class FallStateMachine:
    """Only geometry arms ``suspected``: a horizontal cue plus a fast hip drop, a quick
    change from upright to lying, or a collapse of the box height into lying. The
    event fires once the hip has dropped ``fall_drop_heights`` person heights while
    horizontal, or after lying ``lying_confirm_s``. Recovery never cancels an event."""

    def __init__(self, config: FallConfig) -> None:
        self.config = config
        self.state: State = "normal"
        self._prev: Observation | None = None
        self._baseline_hip: float | None = None
        self._baseline_height: float | None = None
        self._recent: deque[tuple[float, float, float]] = deque()  # (t_s, hip_y, height_px)
        self._window: deque[tuple[float, float, float]] = deque()  # every valid frame, same
        self._last_fast_drop_s = -1.0
        self._last_upright_s = -1.0
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

    def _update_baseline(self, o: Observation) -> None:
        """Highest hip (smallest y) among recent non-horizontal frames, so one frame
        caught mid-descent does not lower the reference. Kept when frames are sparse."""
        self._recent.append((o.t_s, o.hip_y, o.height_px))
        while len(self._recent) > 1 and o.t_s - self._recent[0][0] > self.config.baseline_window_s:
            self._recent.popleft()
        _, self._baseline_hip, self._baseline_height = min(self._recent, key=lambda r: r[1])

    def _collapse(self, o: Observation, lying: bool) -> bool:
        """The box height shrank by ``collapse_ratio`` into lying within ``collapse_window_s``.
        Works without a baseline (a track confirmed mid-fall), and then sets one from
        the tallest recent frame."""
        c = self.config
        self._window.append((o.t_s, o.hip_y, o.height_px))
        while o.t_s - self._window[0][0] > c.collapse_window_s:
            self._window.popleft()
        if not lying or c.collapse_ratio <= 0 or c.collapse_window_s <= 0:
            return False
        _, hip, height = max(self._window, key=lambda r: r[2])
        if o.height_px > (1.0 - c.collapse_ratio) * height:
            return False
        if self._baseline_hip is None:
            self._baseline_hip, self._baseline_height = hip, height
        return True

    def _lost_pose(self, o: Observation) -> StepResult:
        """Without a pose a fall never starts, but a suspicion must still resolve:
        at the timeout a horizontal box counts as the fall, anything else as a near-miss."""
        c = self.config
        if self.state != "suspected" or o.t_s - self._suspected_since_s <= c.suspected_timeout_s:
            return StepResult(self.state)
        self._max_aspect = max(self._max_aspect, o.aspect)
        features = self._features(o.t_s)
        if o.aspect >= c.horizontal_aspect:
            self.state = "fallen"
            self._cooldown_until_s = o.t_s + c.cooldown_s
            return StepResult(self.state, event=True, features={**features, "pose_lost": True})
        self.state = "normal"
        return StepResult(self.state, near_miss=True, features={**features, "pose_lost": True})

    def update(self, o: Observation) -> StepResult:
        c = self.config
        if not o.valid:
            return self._lost_pose(o)
        horizontal = o.torso_angle_deg >= c.horizontal_angle_deg or o.aspect >= c.horizontal_aspect
        lying = o.torso_angle_deg >= c.horizontal_angle_deg and o.aspect >= c.horizontal_aspect
        upright = o.torso_angle_deg <= c.upright_angle_deg and o.aspect <= c.upright_aspect

        if self.state == "normal" and not horizontal:
            self._update_baseline(o)
        collapse = self._collapse(o, lying)
        speed = 0.0
        sparse = False  # frames too far apart to measure the descent speed
        if self._prev is not None and self._baseline_height:
            dt = o.t_s - self._prev.t_s
            if dt > 1e-4:
                speed = (o.hip_y - self._prev.hip_y) / dt / self._baseline_height
                sparse = dt > c.motion_window_s
        self._prev = o
        drop = (
            (o.hip_y - self._baseline_hip) / self._baseline_height
            if self._baseline_hip is not None and self._baseline_height
            else 0.0
        )
        if speed >= c.drop_speed_hps or (sparse and horizontal and drop >= c.fall_drop_heights):
            self._last_fast_drop_s = o.t_s
        self._lying_since_s = (
            (self._lying_since_s if self._lying_since_s >= 0 else o.t_s) if lying else -1.0
        )

        if upright:
            self._last_upright_s = o.t_s
        event = False
        if self.state == "normal":
            fast = (
                self._last_fast_drop_s >= 0 and o.t_s - self._last_fast_drop_s <= c.motion_window_s
            )
            quick = (
                lying
                and self._last_upright_s >= 0
                and o.t_s - self._last_upright_s <= c.transition_s
            )
            sudden = (quick or collapse) and drop >= c.transition_min_drop_heights
            if horizontal and (fast or sudden) and o.t_s >= self._cooldown_until_s:
                self.state = "suspected"
                self._suspected_since_s = self._last_fast_drop_s if fast else o.t_s
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
