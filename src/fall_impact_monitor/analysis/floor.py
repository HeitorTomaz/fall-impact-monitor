"""Person on the floor: lying outside safe zones for too long.

After an observed fall the limit is ``person_down_s``; without one (slow slide,
hidden fall, person entering the view already down) it is ``no_fall_s``. A lying
episode is tied to a place, not to a track id: id switches, tracked/untracked
flicker and single frames that do not look lying keep the episode going; it ends
when nobody has been seen lying there for ``grace_s``. Untracked detections count,
because a person lying at the edge of the frame may be detected only weakly and
never tracked. Children are not exempt.

Confidence (rule-based, uncalibrated): 0.6 at the limit, rising linearly to 1.0 at
twice the limit. Below the alert threshold the episode is stored once as a
near-miss and keeps being evaluated, so it alerts as soon as its score reaches the
threshold.
"""

from dataclasses import dataclass, field

from fall_impact_monitor.analysis.observation import observe
from fall_impact_monitor.analysis.zones import Zone, zone_at
from fall_impact_monitor.api.perception import FrameResult
from fall_impact_monitor.events import Event, EventCategory, make_event
from fall_impact_monitor.perception.tracking import Box, Track, box_iou

_BASE_CONFIDENCE = 0.6
_SAME_PLACE_IOU = 0.3


@dataclass
class _Episode:
    box: Box
    start_s: float
    last_s: float
    track_id: int | None
    identity: str
    track_ids: set[int] = field(default_factory=set)
    fall_observed: bool = False
    near_miss_sent: bool = False
    alerted: bool = False


class FloorAnalyzer:
    """Emits ``person_down`` events: at most one alert per lying episode."""

    def __init__(
        self,
        camera_id: str,
        zones: list[Zone],
        *,
        person_down_s: float = 30.0,
        no_fall_s: float = 60.0,
        threshold: float = 0.5,
        lying_angle_deg: float = 55.0,
        lying_aspect: float = 1.25,
        grace_s: float = 3.0,
        fall_memory_s: float = 120.0,
    ) -> None:
        self._camera_id = camera_id
        self._allowed = [z for z in zones if z.kind == "safe"]
        self._person_down_s = person_down_s
        self._no_fall_s = no_fall_s
        self._threshold = threshold
        self._lying_angle_deg = lying_angle_deg
        self._lying_aspect = lying_aspect
        self._grace_s = grace_s
        self._fall_memory_s = fall_memory_s
        self._episodes: list[_Episode] = []
        self._falls: dict[int, float] = {}

    def note_fall(self, track_id: int, t_s: float) -> None:
        """Record that a possible fall was raised for ``track_id`` at ``t_s``."""
        self._falls[track_id] = t_s

    def _lying(self, box: Box, track: Track | None) -> bool:
        x1, y1, x2, y2 = box
        wide = (x2 - x1) / max(y2 - y1, 1e-6) >= self._lying_aspect
        if track is not None and track.keypoints is not None:
            o = observe(box, track.keypoints, track.keypoint_scores, track.score, 0.0)
            if o.valid:
                return wide and o.torso_angle_deg >= self._lying_angle_deg
        return wide

    def _people_down(
        self, result: FrameResult, frame_size: tuple[int, int]
    ) -> list[tuple[Box, int | None, str]]:
        down: list[tuple[Box, int | None, str]] = [
            (t.box_xyxy, t.track_id, t.identity)
            for t in result.tracks
            if self._lying(t.box_xyxy, t)
        ]
        down += [
            (d.box_xyxy, None, "uncertain")
            for d in result.detections
            if d.track_id is None and self._lying(d.box_xyxy, None)
        ]
        return [p for p in down if zone_at(self._allowed, p[0], frame_size) is None]

    def _episode_for(self, box: Box, track_id: int | None) -> _Episode | None:
        if track_id is not None:
            for episode in self._episodes:
                if track_id in episode.track_ids:
                    return episode
        scored = [(box_iou(e.box, box), e) for e in self._episodes]
        best = max(scored, key=lambda s: s[0], default=None)
        return best[1] if best is not None and best[0] >= _SAME_PLACE_IOU else None

    def update(self, result: FrameResult, frame_size: tuple[int, int]) -> list[Event]:
        if not result.analysed:
            return []
        now_s = result.record.arrival_ns / 1e9
        self._falls = {i: t for i, t in self._falls.items() if now_s - t <= self._fall_memory_s}
        seen: list[_Episode] = []
        for box, track_id, identity in self._people_down(result, frame_size):
            episode = self._episode_for(box, track_id)
            if episode is None:
                episode = _Episode(box, now_s, now_s, track_id, identity)
                self._episodes.append(episode)
            episode.box, episode.last_s, episode.track_id = box, now_s, track_id
            if identity != "stable" or episode.identity != "stable":
                episode.identity = "uncertain"
            if track_id is not None:
                episode.track_ids.add(track_id)
            episode.fall_observed |= any(i in self._falls for i in episode.track_ids)
            if episode not in seen:
                seen.append(episode)
        self._episodes = [e for e in self._episodes if now_s - e.last_s <= self._grace_s]
        events: list[Event] = []
        for episode in seen:
            events += self._evaluate(episode, now_s)
        return events

    def _evaluate(self, episode: _Episode, now_s: float) -> list[Event]:
        limit_s = self._person_down_s if episode.fall_observed else self._no_fall_s
        lying_s = now_s - episode.start_s
        if episode.alerted or lying_s < limit_s:
            return []
        extra = min(1.0, (lying_s - limit_s) / limit_s)
        confidence = _BASE_CONFIDENCE + (1.0 - _BASE_CONFIDENCE) * extra
        if confidence < self._threshold:
            if episode.near_miss_sent:
                return []
            episode.near_miss_sent = True
        else:
            episode.alerted = True
        return [
            make_event(
                camera_id=self._camera_id,
                category=EventCategory.PERSON_DOWN,
                confidence=confidence,
                start_s=episode.start_s,
                end_s=now_s,
                track_id=episode.track_id,
                identity="stable" if episode.identity == "stable" else "uncertain",
                evidence={
                    "fall_observed": episode.fall_observed,
                    "lying_s": round(lying_s, 2),
                    "box": [round(v, 1) for v in episode.box],
                },
                analysed_seqs=[],
                threshold=self._threshold,
            )
        ]
