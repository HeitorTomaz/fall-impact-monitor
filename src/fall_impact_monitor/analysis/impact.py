"""Possible head impact (experimental): fusion of independent evidence per person.

Evidence kinds (strength 0-1, weight):

- ``head_stop`` (0.5): the head moving fast, then stopping or reversing, anywhere but
  inside a safe zone (see ``head.py``);
- ``thump`` (0.35): a sudden sound (see ``audio.py``); it never opens a candidate by
  itself, it only joins one;
- ``reaction`` (0.4): a possible fall (strength = its confidence) or a near-fall
  (half of it) by the same person.

Safe zones mark soft places (bed, sofa); everything else counts as a place where a head
can hit something, so the rules need no zone at all.

A candidate opens with the first person-based evidence and collects evidence for
``window_s`` (thumps from ``window_s`` before it too); a reaction counts from the time
the fall started, and the candidate waits a few more seconds for the fall rule to
confirm it. Then:

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
    "head_stop": 0.5,
    "thump": 0.35,
    "reaction": 0.4,
}
_NEAR_FALL_FACTOR = 0.5
# A fall is confirmed up to ~3 s after it starts and a near-fall only after the person is
# up again: candidates stay open this much longer so the reaction can join them.
_REACTION_LATENCY_S = 4.0
_FORGET_S = 30.0  # a person not seen for this long is forgotten
_MAX_SEQS = 200
# Full-rate analysis budget per camera: at most this many seconds within any window. Reading
# is synchronous, so sustained full rate on a slow CPU (children playing) would put the
# camera behind real time and delay every alert.
_ATTENTION_BUDGET_S = 15.0
_ATTENTION_BUDGET_WINDOW_S = 60.0


@dataclass
class _Candidate:
    track_id: int
    identity: str
    opened_s: float
    strengths: dict[str, float] = field(default_factory=dict)
    times: list[float] = field(default_factory=list)
    seqs: list[int] = field(default_factory=list)

    def add(self, kind: str, t_s: float, strength: float, seq: int | None = None) -> None:
        self.strengths[kind] = max(self.strengths.get(kind, 0.0), strength)
        self.times.append(t_s)
        if seq is not None and len(self.seqs) < _MAX_SEQS:
            self.seqs.append(seq)


@dataclass
class _Person:
    tracker: HeadTracker
    last: HeadSample | None = None
    last_seen_s: float = 0.0


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
        self._safe = [z for z in zones if z.kind == "safe"]
        self._threshold = threshold
        self._min_evidence = min_evidence
        self._window_s = window_s
        self._attention_s = attention_s
        self._config = config or HeadConfig()
        self._people: dict[int, _Person] = {}
        self._candidates: dict[int, _Candidate] = {}
        self._thumps: deque[Thump] = deque()  # appended from the audio thread
        self._attention_until: float | None = None
        self._attention_spans: deque[list[float]] = deque()  # [start, end] seconds
        self._frame_size = (1, 1)

    def add_thumps(self, thumps: list[Thump]) -> None:
        """Thread-safe (deque append): called by the audio reader."""
        self._thumps.extend(thumps)

    def attention_until(self) -> float | None:
        return self._attention_until

    def people_count(self) -> int:
        return len(self._people)

    def _request_attention(self, now_s: float) -> None:
        """Full rate for ``attention_s``, within the budget per window."""
        window_start = now_s - _ATTENTION_BUDGET_WINDOW_S
        while self._attention_spans and self._attention_spans[0][1] <= window_start:
            self._attention_spans.popleft()
        used = sum(
            max(0.0, min(end, now_s) - max(start, window_start))
            for start, end in self._attention_spans
        )
        if used >= _ATTENTION_BUDGET_S:
            return
        until = now_s + self._attention_s
        if self._attention_spans and self._attention_spans[-1][1] >= now_s:
            self._attention_spans[-1][1] = until
        else:
            self._attention_spans.append([now_s, until])
        self._attention_until = until

    def _thump_memory_s(self) -> float:
        return 2 * self._window_s + _REACTION_LATENCY_S + 2.0

    def _in_safe_zone(self, x: float, y: float) -> bool:
        w, h = self._frame_size
        return any(zone.contains(x / w, y / h) for zone in self._safe)

    def _candidate(self, track_id: int, identity: str, now_s: float) -> _Candidate:
        candidate = self._candidates.get(track_id)
        if candidate is None:
            candidate = _Candidate(track_id, identity, now_s)
            self._candidates[track_id] = candidate
        return candidate

    def update(
        self, result: FrameResult, frame_size: tuple[int, int], fall_events: list[Event]
    ) -> list[Event]:
        if not result.analysed:
            return []
        self._frame_size = frame_size
        now_s = result.record.arrival_ns / 1e9
        valid = result.record.quality is TimestampQuality.OK
        seq = result.record.seq
        for track in result.tracks:
            person = self._people.setdefault(track.track_id, _Person(HeadTracker(self._config)))
            person.last_seen_s = now_s
            point = head_point(track)
            if point is None:
                continue
            height_px = track.box_xyxy[3] - track.box_xyxy[1]
            sample = person.tracker.update(now_s, point, height_px, valid_time=valid)
            if sample is None or self._in_safe_zone(sample.x, sample.y):
                person.last = sample
                continue
            # Full rate only at the evidence speed: walking (about 0.6 person heights per
            # second) must not keep a camera at full rate.
            if sample.speed_hps >= self._config.min_speed_hps:
                self._request_attention(now_s)
            if person.last is not None:
                strength = head_stop(person.last, sample, self._config)
                if strength is not None:
                    self._candidate(track.track_id, track.identity, now_s).add(
                        "head_stop", now_s, strength, seq
                    )
            person.last = sample
        self._people = {k: p for k, p in self._people.items() if now_s - p.last_seen_s <= _FORGET_S}
        for fall in fall_events:
            if fall.category is not EventCategory.POSSIBLE_FALL or fall.track_id is None:
                continue
            # A fall that ended in a safe zone (lying down in bed) is no more than a
            # near-fall here.
            near_fall = bool(fall.evidence.get("near_fall") or fall.evidence.get("safe_zone"))
            strength = fall.confidence * (_NEAR_FALL_FACTOR if near_fall else 1.0)
            candidate = self._candidate(fall.track_id, fall.identity, fall.start_s)
            candidate.add("reaction", fall.start_s, strength, seq)
        return self._close(now_s)

    def _close(self, now_s: float) -> list[Event]:
        while self._thumps and now_s - self._thumps[0].t_s > self._thump_memory_s():
            self._thumps.popleft()
        events = []
        for track_id, candidate in list(self._candidates.items()):
            if now_s - candidate.opened_s < self._window_s + _REACTION_LATENCY_S:
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
            analysed_seqs=list(candidate.seqs),
            threshold=self._threshold,
        )
        return event if enough else dataclasses.replace(event, near_miss=True)
