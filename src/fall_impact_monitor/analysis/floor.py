"""Person on the floor: lying outside allowed zones for too long.

After an observed fall the limit is ``person_down_s``; without one (slow slide,
hidden fall, person entering the view already down) it is ``no_fall_s``. Untracked
detections count too, by persistence at the same place, because a person lying at
the edge of the frame may be detected only weakly and never tracked. Children are
not exempt. Confidence (rule-based, uncalibrated): 0.6 at the limit, rising
linearly to 1.0 at twice the limit; a person-down event is never a near-miss at
the default threshold.
"""

from dataclasses import dataclass

from fall_impact_monitor.analysis.observation import observe
from fall_impact_monitor.analysis.zones import Zone
from fall_impact_monitor.api.perception import FrameResult
from fall_impact_monitor.events import Event, EventCategory, make_event
from fall_impact_monitor.perception.tracking import Track, box_iou

Box = tuple[float, float, float, float]

_BASE_CONFIDENCE = 0.6
_SAME_PLACE_IOU = 0.3


@dataclass
class _Episode:
    box: Box
    start_s: float
    last_s: float
    fired: bool = False


class FloorAnalyzer:
    """Emits ``person_down`` events, once per lying episode."""

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
        self._allowed = [z for z in zones if z.kind == "allowed_lying"]
        self._person_down_s = person_down_s
        self._no_fall_s = no_fall_s
        self._threshold = threshold
        self._lying_angle_deg = lying_angle_deg
        self._lying_aspect = lying_aspect
        self._grace_s = grace_s
        self._fall_memory_s = fall_memory_s
        self._tracked: dict[int, _Episode] = {}
        self._untracked: list[_Episode] = []
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

    def _allowed_place(self, box: Box, frame_size: tuple[int, int]) -> bool:
        cx = (box[0] + box[2]) / 2 / frame_size[0]
        cy = (box[1] + box[3]) / 2 / frame_size[1]
        return any(zone.contains(cx, cy) for zone in self._allowed)

    def _down(self, box: Box, track: Track | None, frame_size: tuple[int, int]) -> bool:
        return self._lying(box, track) and not self._allowed_place(box, frame_size)

    def update(self, result: FrameResult, frame_size: tuple[int, int]) -> list[Event]:
        if not result.analysed:
            return []
        now_s = result.record.arrival_ns / 1e9
        events: list[Event] = []
        for track in result.tracks:
            if not self._down(track.box_xyxy, track, frame_size):
                self._tracked.pop(track.track_id, None)
                continue
            episode = self._tracked.setdefault(
                track.track_id, _Episode(track.box_xyxy, now_s, now_s)
            )
            episode.box, episode.last_s = track.box_xyxy, now_s
            fall_s = self._falls.get(track.track_id)
            fall_observed = fall_s is not None and now_s - fall_s <= self._fall_memory_s
            limit_s = self._person_down_s if fall_observed else self._no_fall_s
            identity = "stable" if track.identity == "stable" else "uncertain"
            events += self._fire(episode, now_s, limit_s, track.track_id, identity, fall_observed)
        self._tracked = {
            i: e for i, e in self._tracked.items() if now_s - e.last_s <= self._grace_s
        }

        for person in result.detections:
            if person.track_id is not None or not self._down(person.box_xyxy, None, frame_size):
                continue
            spot = next(
                (e for e in self._untracked if box_iou(e.box, person.box_xyxy) >= _SAME_PLACE_IOU),
                None,
            )
            if spot is None:
                spot = _Episode(person.box_xyxy, now_s, now_s)
                self._untracked.append(spot)
            spot.box, spot.last_s = person.box_xyxy, now_s
            events += self._fire(spot, now_s, self._no_fall_s, None, "uncertain", False)
        self._untracked = [e for e in self._untracked if now_s - e.last_s <= self._grace_s]
        return events

    def _fire(
        self,
        episode: _Episode,
        now_s: float,
        limit_s: float,
        track_id: int | None,
        identity: str,
        fall_observed: bool,
    ) -> list[Event]:
        lying_s = now_s - episode.start_s
        if episode.fired or lying_s < limit_s:
            return []
        episode.fired = True
        extra = min(1.0, (lying_s - limit_s) / limit_s) if limit_s > 0 else 1.0
        return [
            make_event(
                camera_id=self._camera_id,
                category=EventCategory.PERSON_DOWN,
                confidence=_BASE_CONFIDENCE + (1.0 - _BASE_CONFIDENCE) * extra,
                start_s=episode.start_s,
                end_s=now_s,
                track_id=track_id,
                identity="stable" if identity == "stable" else "uncertain",
                evidence={
                    "fall_observed": fall_observed,
                    "lying_s": round(lying_s, 2),
                    "box": [round(v, 1) for v in episode.box],
                },
                analysed_seqs=[],
                threshold=self._threshold,
            )
        ]
