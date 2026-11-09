"""Multi-person tracking (OC-SORT) with an explicit identity-certainty flag.

Identity is ``uncertain`` while a track is young and while (and shortly after)
its box overlaps another person's: an identity swap is most likely then, so
downstream rules must not trust per-person history (an uncertain person is
treated as an adult).
"""

import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import supervision as sv
from numpy.typing import NDArray
from trackers import OCSORTTracker

from fall_impact_monitor.perception.detector import Detection
from fall_impact_monitor.perception.pose import Pose

Identity = Literal["stable", "uncertain"]
Box = tuple[float, float, float, float]


@dataclass(frozen=True)
class Track:
    track_id: int
    box_xyxy: Box
    score: float
    keypoints: NDArray[np.float32] | None
    keypoint_scores: NDArray[np.float32] | None
    identity: Identity
    age_frames: int
    detection_index: int  # index of the detection this track was matched to in this frame


def box_iou(a: Box, b: Box) -> float:
    inter_w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    inter_h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = inter_w * inter_h
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


class PersonTracker:
    """Wraps OC-SORT (re-identification disabled) and fed with absolute timestamps."""

    def __init__(
        self,
        *,
        overlap_iou: float = 0.1,
        uncertain_hold_s: float = 2.0,
        min_age_frames: int = 5,
        frame_rate: float = 5.0,
        new_track_min_score: float = 0.3,
        track_lost_s: float = 3.0,
    ) -> None:
        # With timestamps, the library converts the lost-track buffer from frames to
        # seconds at 30 frames per second, so the buffer is expressed in 1/30 s units.
        self._tracker = OCSORTTracker(
            frame_rate=frame_rate,
            lost_track_buffer=max(1, round(track_lost_s * 30)),
            minimum_consecutive_frames=3,
            high_conf_det_threshold=new_track_min_score,
        )
        self._overlap_iou = overlap_iou
        self._uncertain_hold_s = uncertain_hold_s
        self._min_age_frames = min_age_frames
        self._age: dict[int, int] = {}
        self._uncertain_until_s: dict[int, float] = {}
        self._last_seen_s: dict[int, float] = {}
        self._forget_after_s = 2 * track_lost_s + uncertain_hold_s

    def update(
        self, detections: Sequence[Detection], poses: Sequence[Pose | None], timestamp_s: float
    ) -> list[Track]:
        """Associate this frame's detections (and their poses) with tracks."""
        if detections:
            batch = sv.Detections(
                xyxy=np.array([d.box_xyxy for d in detections], np.float64),
                confidence=np.array([d.score for d in detections], np.float64),
                class_id=np.zeros(len(detections), int),
                data={"det_index": np.arange(len(detections))},
            )
        else:
            batch = sv.Detections.empty()
        with warnings.catch_warnings():
            # Repeated timestamps are expected from some cameras; the tracker skips them.
            warnings.simplefilter("ignore", UserWarning)
            result = self._tracker.update(batch, timestamp=timestamp_s)
        if not detections or result.tracker_id is None:
            return []

        confirmed: list[tuple[int, int]] = [
            (int(track_id), int(det_index))
            for track_id, det_index in zip(result.tracker_id, result.data["det_index"], strict=True)
            if track_id >= 0
        ]
        for track_id, _ in confirmed:
            self._age[track_id] = self._age.get(track_id, 0) + 1
        for track_id, det_index in confirmed:
            # Any other detection (tracked or not, weak or strong) overlapping this
            # person makes an identity swap possible.
            for other_index, other in enumerate(detections):
                if (
                    other_index != det_index
                    and box_iou(detections[det_index].box_xyxy, other.box_xyxy) > self._overlap_iou
                ):
                    self._uncertain_until_s[track_id] = timestamp_s + self._uncertain_hold_s
                    break
        for track_id, _ in confirmed:
            self._last_seen_s[track_id] = timestamp_s
        forget_before_s = timestamp_s - self._forget_after_s
        for stale in [i for i, seen in self._last_seen_s.items() if seen < forget_before_s]:
            self._last_seen_s.pop(stale)
            self._age.pop(stale, None)
            self._uncertain_until_s.pop(stale, None)

        tracks = []
        for track_id, det_index in confirmed:
            age = self._age[track_id]
            recently_overlapped = timestamp_s <= self._uncertain_until_s.get(track_id, -1.0)
            identity: Identity = (
                "uncertain" if age < self._min_age_frames or recently_overlapped else "stable"
            )
            pose = poses[det_index]
            detection = detections[det_index]
            tracks.append(
                Track(
                    track_id=track_id,
                    box_xyxy=detection.box_xyxy,
                    score=detection.score,
                    keypoints=pose.keypoints if pose else None,
                    keypoint_scores=pose.scores if pose else None,
                    identity=identity,
                    age_frames=age,
                    detection_index=det_index,
                )
            )
        return tracks
