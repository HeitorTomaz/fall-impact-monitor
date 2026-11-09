"""Fall analysis per camera: one state machine per tracked person, scored events.

Confidence (rule-based, uncalibrated until calibrated on site):

    evidence = 0.30 * drop + 0.25 * speed + 0.30 * posture + 0.15 * lying
      drop    = min(1, max_drop_heights / (2 * fall_drop_heights))
      speed   = min(1, max_drop_speed_hps / (2 * drop_speed_hps))
      posture = 0.5 * clip((max_torso_angle_deg - 35) / 45) + 0.5 * clip(max_aspect - 1)
      lying   = min(1, lying_s / 2)
    descent rule met:           0.3 + 0.7 * evidence
    track lost while falling:   0.4 + 0.7 * evidence
    suspicion without the rule:       0.7 * evidence

Every event is a near-miss only when its score is below the alert threshold.
"""

from dataclasses import dataclass
from typing import Any

from fall_impact_monitor.analysis.fall_state import FallConfig, FallStateMachine
from fall_impact_monitor.analysis.observation import observe
from fall_impact_monitor.api.perception import FrameResult
from fall_impact_monitor.events import Event, EventCategory, make_event
from fall_impact_monitor.perception.tracking import Box, Track

_RULE_MET_BASE = 0.3
_TRACK_LOST_BASE = 0.4
_EVIDENCE_WEIGHT = 0.7
_MAX_SEQS = 200  # analysed frames kept per event


def _clip(value: float) -> float:
    return min(1.0, max(0.0, value))


def fall_confidence(
    features: dict[str, Any],
    *,
    rule_met: bool,
    config: FallConfig | None = None,
    base: float | None = None,
) -> float:
    config = config or FallConfig()
    drop = _clip(features["max_drop_heights"] / (2 * config.fall_drop_heights))
    speed = _clip(features["max_drop_speed_hps"] / (2 * config.drop_speed_hps))
    posture = 0.5 * _clip((features["max_torso_angle_deg"] - 35.0) / 45.0) + 0.5 * _clip(
        features["max_aspect"] - 1.0
    )
    lying = _clip(features["lying_s"] / 2.0)
    evidence = 0.30 * drop + 0.25 * speed + 0.30 * posture + 0.15 * lying
    offset = base if base is not None else (_RULE_MET_BASE if rule_met else 0.0)
    return _clip(offset + _EVIDENCE_WEIGHT * evidence)


@dataclass
class _PersonState:
    machine: FallStateMachine
    last_seen_s: float
    last_box: Box
    identity: str
    seqs: list[int]
    suspected_start_s: float | None = None
    last_features: dict[str, Any] | None = None


class FallAnalyzer:
    """Emits ``possible_fall`` events (and near-misses) from analysed frames."""

    def __init__(
        self,
        camera_id: str,
        config: FallConfig | None = None,
        *,
        threshold: float = 0.5,
        track_lost_s: float = 1.0,
        forget_s: float = 30.0,
        handover_s: float = 3.0,
    ) -> None:
        self._camera_id = camera_id
        self._config = config or FallConfig()
        self._threshold = threshold
        self._track_lost_s = track_lost_s
        self._forget_s = forget_s
        self._handover_s = handover_s
        self._people: dict[int, _PersonState] = {}

    def update(self, result: FrameResult) -> list[Event]:
        if not result.analysed:
            return []
        now_s = result.record.arrival_ns / 1e9
        seq = result.record.seq
        events: list[Event] = []
        visible = {t.track_id for t in result.tracks}
        for track in result.tracks:
            events += self._update_track(track, now_s, seq, visible)
        events += self._check_lost(now_s, visible)
        return events

    def _event(
        self,
        person: _PersonState,
        track_id: int,
        now_s: float,
        confidence: float,
        evidence: dict[str, Any],
    ) -> Event:
        return make_event(
            camera_id=self._camera_id,
            category=EventCategory.POSSIBLE_FALL,
            confidence=confidence,
            start_s=min(person.suspected_start_s or now_s, now_s),
            end_s=now_s,
            track_id=track_id,
            identity="uncertain" if person.identity == "uncertain" else "stable",
            evidence=evidence,
            analysed_seqs=list(person.seqs),
            threshold=self._threshold,
        )

    def _handover(self, box: Box, now_s: float, visible: set[int]) -> _PersonState | None:
        """A new track id near where a person was lost moments ago (the tracker often
        re-identifies a falling person) continues that person's state machine."""
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        best: tuple[float, int] | None = None
        for track_id, person in self._people.items():
            if track_id in visible or now_s - person.last_seen_s > self._handover_s:
                continue
            x1, y1, x2, y2 = person.last_box
            reach = y2 - y1  # one person height around the last box
            if x1 - reach <= cx <= x2 + reach and y1 - reach <= cy <= y2 + reach:
                distance = abs(cx - (x1 + x2) / 2) + abs(cy - (y1 + y2) / 2)
                if best is None or distance < best[0]:
                    best = (distance, track_id)
        return self._people.pop(best[1]) if best is not None else None

    def _update_track(self, track: Track, now_s: float, seq: int, visible: set[int]) -> list[Event]:
        person = self._people.get(track.track_id)
        if person is None:
            person = self._handover(track.box_xyxy, now_s, visible) or _PersonState(
                FallStateMachine(self._config), now_s, track.box_xyxy, track.identity, []
            )
            self._people[track.track_id] = person
        person.last_seen_s = now_s
        person.last_box = track.box_xyxy
        person.identity = track.identity
        before = person.machine.state
        step = person.machine.update(
            observe(track.box_xyxy, track.keypoints, track.keypoint_scores, track.score, now_s)
        )
        if before == "normal" and step.state == "suspected":
            person.suspected_start_s = now_s
            person.seqs = []
        if step.state == "suspected" or step.event or step.near_miss:
            person.seqs = [*person.seqs[-(_MAX_SEQS - 1) :], seq]
            person.last_features = step.features
        if step.event:
            confidence = fall_confidence(step.features, rule_met=True, config=self._config)
            return [self._event(person, track.track_id, now_s, confidence, dict(step.features))]
        if step.near_miss:
            confidence = fall_confidence(step.features, rule_met=False, config=self._config)
            evidence = {**step.features, "near_fall": True}
            return [self._event(person, track.track_id, now_s, confidence, evidence)]
        return []

    def _check_lost(self, now_s: float, visible: set[int]) -> list[Event]:
        events: list[Event] = []
        for track_id, person in list(self._people.items()):
            if track_id in visible:
                continue
            absent_s = now_s - person.last_seen_s
            if person.machine.state == "suspected" and absent_s >= self._track_lost_s:
                features = person.last_features or {}
                confidence = (
                    fall_confidence(
                        features, rule_met=False, config=self._config, base=_TRACK_LOST_BASE
                    )
                    if features
                    else _TRACK_LOST_BASE
                )
                evidence = {**features, "track_lost": True, "absent_s": round(absent_s, 2)}
                events.append(self._event(person, track_id, now_s, confidence, evidence))
                del self._people[track_id]
            elif absent_s >= self._forget_s:
                del self._people[track_id]
        return events
