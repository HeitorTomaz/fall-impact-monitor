"""Possible head impact (experimental): fusion of independent evidence per person.

Evidence kinds (strength 0-1, weight):

- ``head_stop_near_hazard`` (0.5): the head moving fast, then stopping or reversing, near
  a hazard zone (see ``head.py``);
- ``thump`` (0.35): a sudden sound (see ``audio.py``); it never opens a candidate by
  itself, it only joins one;
- ``reaction`` (0.4): a possible fall (strength = its confidence) or a near-fall
  (half of it) by the same person;
- ``head_in_hazard_zone`` (0.45): the person's head was inside a hazard zone when a
  possible fall was raised.

A candidate opens with the first person-based evidence and collects evidence for
``window_s`` (thumps from ``window_s`` before it too). Then:

    confidence = 1 - prod(1 - weight_k * strength_k)   over the kinds present

It alerts only with at least ``min_evidence`` kinds **and** a confidence at or above the
alert threshold; otherwise it is stored as a near-miss. Scores are rule-based and
uncalibrated. One camera cannot see a contact: this is never a measured impact.
"""

import dataclasses
from collections import deque
from dataclasses import dataclass, field

from fall_impact_monitor.analysis.audio import Thump
from fall_impact_monitor.analysis.head import (
    HeadConfig,
    HeadSample,
    HeadTracker,
    head_point,
    head_stop,
)
from fall_impact_monitor.analysis.zones import Zone
from fall_impact_monitor.api.perception import FrameResult
from fall_impact_monitor.capture import TimestampQuality
from fall_impact_monitor.events import Event, EventCategory, make_event

WEIGHTS = {
    "head_stop_near_hazard": 0.5,
    "thump": 0.35,
    "reaction": 0.4,
    "head_in_hazard_zone": 0.45,
}
_THUMP_MEMORY_S = 10.0
_NEAR_FALL_FACTOR = 0.5


@dataclass
class _Candidate:
    track_id: int
    identity: str
    opened_s: float
    strengths: dict[str, float] = field(default_factory=dict)
    times: list[float] = field(default_factory=list)

    def add(self, kind: str, t_s: float, strength: float) -> None:
        self.strengths[kind] = max(self.strengths.get(kind, 0.0), strength)
        self.times.append(t_s)


@dataclass
class _Person:
    tracker: HeadTracker
    last: HeadSample | None = None
    point: tuple[float, float] | None = None
    height_px: float = 0.0


class HeadImpactAnalyzer:
    def __init__(
        self,
        camera_id: str,
        zones: list[Zone],
        *,
        threshold: float = 0.5,
        min_evidence: int = 2,
        window_s: float = 3.0,
        attention_s: float = 2.0,
        config: HeadConfig | None = None,
    ) -> None:
        self._camera_id = camera_id
        self._hazards = [z for z in zones if z.kind == "hazard"]
        self._threshold = threshold
        self._min_evidence = min_evidence
        self._window_s = window_s
        self._attention_s = attention_s
        self._config = config or HeadConfig()
        self._people: dict[int, _Person] = {}
        self._candidates: dict[int, _Candidate] = {}
        self._thumps: deque[Thump] = deque()  # appended from the audio thread
        self._attention_until: float | None = None
        self._frame_size = (1, 1)

    def add_thumps(self, thumps: list[Thump]) -> None:
        """Thread-safe (deque append): called by the audio reader."""
        self._thumps.extend(thumps)

    def attention_until(self) -> float | None:
        return self._attention_until

    def _near_hazard(self, point: tuple[float, float], height_px: float) -> bool:
        margin = self._config.head_size_heights * height_px
        w, h = self._frame_size
        return any(
            zone.contains((point[0] + dx) / w, (point[1] + dy) / h)
            for zone in self._hazards
            for dx, dy in ((0, 0), (margin, 0), (-margin, 0), (0, margin), (0, -margin))
        )

    def _candidate(self, track_id: int, identity: str, now_s: float) -> _Candidate:
        candidate = self._candidates.get(track_id)
        if candidate is None:
            candidate = _Candidate(track_id, identity, now_s)
            self._candidates[track_id] = candidate
        return candidate

    def update(
        self, result: FrameResult, frame_size: tuple[int, int], fall_events: list[Event]
    ) -> list[Event]:
        if not result.analysed or not self._hazards:
            return []
        self._frame_size = frame_size
        now_s = result.record.arrival_ns / 1e9
        valid = result.record.quality is TimestampQuality.OK
        for track in result.tracks:
            person = self._people.setdefault(track.track_id, _Person(HeadTracker(self._config)))
            point = head_point(track)
            if point is None:
                continue
            height_px = track.box_xyxy[3] - track.box_xyxy[1]
            person.point, person.height_px = point, height_px
            if self._near_hazard(point, height_px):
                self._attention_until = now_s + self._attention_s
            sample = person.tracker.update(now_s, point, height_px, valid_time=valid)
            if person.last is not None and sample is not None:
                strength = head_stop(person.last, sample, self._hazards, frame_size, self._config)
                if strength is not None:
                    self._candidate(track.track_id, track.identity, now_s).add(
                        "head_stop_near_hazard", now_s, strength
                    )
            person.last = sample
        for fall in fall_events:
            if fall.category is not EventCategory.POSSIBLE_FALL or fall.track_id is None:
                continue
            near_fall = bool(fall.evidence.get("near_fall"))
            strength = fall.confidence * (_NEAR_FALL_FACTOR if near_fall else 1.0)
            candidate = self._candidate(fall.track_id, fall.identity, now_s)
            candidate.add("reaction", now_s, strength)
            faller = self._people.get(fall.track_id)
            if (
                not near_fall
                and faller is not None
                and faller.point is not None
                and self._near_hazard(faller.point, faller.height_px)
            ):
                candidate.add("head_in_hazard_zone", now_s, 1.0)
        return self._close(now_s)

    def _close(self, now_s: float) -> list[Event]:
        while self._thumps and now_s - self._thumps[0].t_s > _THUMP_MEMORY_S:
            self._thumps.popleft()
        events = []
        for track_id, candidate in list(self._candidates.items()):
            if now_s - candidate.opened_s < self._window_s:
                continue
            del self._candidates[track_id]
            start = candidate.opened_s - self._window_s
            end = candidate.opened_s + self._window_s
            for thump in list(self._thumps):
                if start <= thump.t_s <= end:
                    candidate.add("thump", thump.t_s, thump.strength)
            events.append(self._event(candidate))
        return events

    def _event(self, candidate: _Candidate) -> Event:
        remaining = 1.0
        for kind, strength in candidate.strengths.items():
            remaining *= 1.0 - WEIGHTS[kind] * strength
        confidence = 1.0 - remaining
        kinds = sorted(candidate.strengths)
        enough = len(kinds) >= self._min_evidence
        event = make_event(
            camera_id=self._camera_id,
            category=EventCategory.POSSIBLE_HEAD_IMPACT,
            confidence=confidence,
            start_s=min(candidate.times),
            end_s=max(candidate.times),
            track_id=candidate.track_id,
            identity="uncertain" if candidate.identity == "uncertain" else "stable",
            evidence={
                "kinds": kinds,
                "strengths": {k: round(v, 3) for k, v in candidate.strengths.items()},
                "experimental": True,
                **({} if enough else {"reason": f"fewer than {self._min_evidence} kinds"}),
            },
            analysed_seqs=[],
            threshold=self._threshold,
        )
        return event if enough else dataclasses.replace(event, near_miss=True)
