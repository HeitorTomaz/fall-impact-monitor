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

Every event is a near-miss when its score is below the alert threshold.

Allowed-lying zones (bed, sofa, a mattress on the floor): a possible fall whose person's
box centre is inside one when the event fires is a near-miss (lying down in bed), and
the state machine returns to normal so that a later fall can still be raised. A person
who was lying inside a zone and is then seen lying outside every zone (from the bed to
the floor) arms a suspicion, even without a measurable drop.
"""

import dataclasses
from dataclasses import dataclass
from typing import Any

from fall_impact_monitor.analysis.fall_state import FallConfig, FallStateMachine
from fall_impact_monitor.analysis.observation import Observation, observe
from fall_impact_monitor.analysis.zones import Zone, zone_at
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
    zone_name: str | None = None  # allowed zone the person was last seen lying in
    in_zone_s: float = -1.0
    outside_since_s: float | None = None  # lying outside every zone since
    left_zone: str | None = None  # zone left while lying, for the next event


class FallAnalyzer:
    """Emits ``possible_fall`` events (and near-misses) from analysed frames."""

    def __init__(
        self,
        camera_id: str,
        config: FallConfig | None = None,
        *,
        zones: list[Zone] | None = None,
        threshold: float = 0.5,
        track_lost_s: float = 1.0,
        forget_s: float = 30.0,
        handover_s: float = 3.0,
        zone_exit_s: float = 3.0,
    ) -> None:
        self._camera_id = camera_id
        self._allowed = [z for z in zones or [] if z.kind == "allowed_lying"]
        self._zone_exit_s = zone_exit_s
        self._frame_size = (1, 1)
        self._config = config or FallConfig()
        self._threshold = threshold
        self._track_lost_s = track_lost_s
        self._forget_s = forget_s
        self._handover_s = handover_s
        self._people: dict[int, _PersonState] = {}

    def update(self, result: FrameResult, frame_size: tuple[int, int]) -> list[Event]:
        if not result.analysed:
            return []
        self._frame_size = frame_size
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

    @staticmethod
    def _in_zone(event: Event, zone: Zone) -> Event:
        """Lying down in an allowed zone is not notified; it is kept as a near-miss."""
        evidence = {**event.evidence, "allowed_zone": zone.name}
        return dataclasses.replace(event, near_miss=True, evidence=evidence)

    def _left_zone(
        self, person: _PersonState, o: Observation, zone: Zone | None, now_s: float
    ) -> str | None:
        """The name of the allowed zone a lying person has just left: seen lying inside
        it, then lying outside every zone within ``zone_exit_s`` and for at least
        ``lying_confirm_s`` (so a box flickering over the zone edge does not count)."""
        c = self._config
        lying = o.aspect >= c.horizontal_aspect and (
            not o.valid or o.torso_angle_deg >= c.horizontal_angle_deg
        )
        if not lying:
            person.outside_since_s = None
            return None
        if zone is not None:
            person.zone_name, person.in_zone_s, person.outside_since_s = zone.name, now_s, None
            return None
        if person.outside_since_s is None:
            if person.zone_name is None or now_s - person.in_zone_s > self._zone_exit_s:
                return None
            person.outside_since_s = now_s
        if now_s - person.outside_since_s < c.lying_confirm_s:
            return None
        left, person.zone_name, person.outside_since_s = person.zone_name, None, None
        return left

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
        o = observe(track.box_xyxy, track.keypoints, track.keypoint_scores, track.score, now_s)
        zone = zone_at(self._allowed, track.box_xyxy, self._frame_size)
        left = self._left_zone(person, o, zone, now_s)
        if left is not None and person.machine.state == "normal":
            person.machine.arm(now_s)
            person.left_zone = left
        step = person.machine.update(o)
        if before == "normal" and step.state == "suspected":
            person.suspected_start_s = now_s
            person.seqs = []
        if step.state == "suspected" or step.event or step.near_miss:
            person.seqs = [*person.seqs[-(_MAX_SEQS - 1) :], seq]
            person.last_features = step.features
        if step.event:
            confidence = fall_confidence(step.features, rule_met=True, config=self._config)
            evidence = dict(step.features)
            if person.left_zone is not None:
                evidence["left_allowed_zone"] = person.left_zone
            person.left_zone = person.zone_name = None
            event = self._event(person, track.track_id, now_s, confidence, evidence)
            if zone is not None:
                person.machine.release()
                return [self._in_zone(event, zone)]
            return [event]
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
                event = self._event(person, track_id, now_s, confidence, evidence)
                zone = zone_at(self._allowed, person.last_box, self._frame_size)
                events.append(event if zone is None else self._in_zone(event, zone))
                del self._people[track_id]
            elif absent_s >= self._forget_s:
                del self._people[track_id]
        return events
