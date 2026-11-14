"""Perception pipeline: frames in, tracked people (box, pose, identity) out."""

from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import combinations
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from fall_impact_monitor.capture.timing import FrameRecord
from fall_impact_monitor.perception.cadence import CadencePolicy, RateMonitor, SceneState
from fall_impact_monitor.perception.detector import Detection
from fall_impact_monitor.perception.follow import (
    MIN_FOLLOW_IOU,
    Followed,
    apply_margins,
    box_margins,
    clip_box,
    keypoint_box,
    search_box,
)
from fall_impact_monitor.perception.pose import Box, Pose
from fall_impact_monitor.perception.tracking import PersonTracker, Track, box_iou

_CONFIRMATION_MIN_FPS = 1.0  # unused there: the rate is never judged
_EPSILON_S = 1e-6


class Detector(Protocol):
    def detect(self, image_bgr: NDArray[np.uint8]) -> list[Detection]: ...


class PoseModel(Protocol):
    def estimate(self, image_bgr: NDArray[np.uint8], boxes: Sequence[Box]) -> list[Pose | None]: ...


@dataclass(frozen=True)
class DetectedPerson:
    """One person detection of an analysed frame, with its pose and track (if any)."""

    box_xyxy: Box
    score: float
    pose: Pose | None
    track_id: int | None


@dataclass(frozen=True)
class FrameResult:
    """Outcome of one frame.

    ``detections`` holds every person detection of an analysed frame, including
    weak ones that are not (yet) part of a confirmed track: a person lying at the
    edge of the frame may only ever be detected weakly, and must stay visible
    to the rules that look for people on the floor.
    """

    record: FrameRecord
    tracks: list[Track]
    analysed: bool
    detections: list[DetectedPerson] = field(default_factory=list)


class PerceptionPipeline:
    """Decides which frames to analyse and runs detection, pose and tracking on them.

    Scheduling and tracking use the monotonic arrival time, which always
    increases; source PTS may repeat or jump and is kept only in the records.

    With ``detector_max_fps`` above 0, the person detector runs at most that often and
    the people it found are followed by the pose estimator on the analysed frames in
    between (``perception.follow``). The detector runs at once instead when a pose is
    lost (a person leaving the image), a box changes a lot (a fall) or overlaps another
    person's, or a person had no usable pose. 0 runs it on every analysed frame.
    """

    def __init__(
        self,
        detector: Detector,
        pose: PoseModel,
        tracker: PersonTracker,
        cadence: CadencePolicy,
        rate_monitor: RateMonitor,
        *,
        person_hold_s: float = 3.0,
        detector_max_fps: float = 0.0,
    ) -> None:
        self._detector = detector
        self._pose = pose
        self._tracker = tracker
        self._cadence = cadence
        self._rate = rate_monitor
        self._last_analysed_s: float | None = None
        self._last_person_s: float | None = None
        self._person_hold_s = person_hold_s
        self._attention_until_s = -1.0
        self._detector_interval_s = 1.0 / detector_max_fps if detector_max_fps > 0 else 0.0
        self._last_detector_s: float | None = None
        self._followed: list[Followed] = []

    def request_attention(self, until_s: float) -> None:
        """Analyse every frame until ``until_s`` (arrival seconds)."""
        self._attention_until_s = max(self._attention_until_s, until_s)

    def state(self, now_s: float) -> SceneState:
        if now_s <= self._attention_until_s:
            return "attention"
        recent_person = (
            self._last_person_s is not None and now_s - self._last_person_s <= self._person_hold_s
        )
        return "person" if recent_person else "empty"

    def process(self, frame_bgr: NDArray[np.uint8], record: FrameRecord) -> FrameResult:
        now_s = record.arrival_ns / 1e9
        interval_s = self._cadence.interval_s(self.state(now_s))
        if (
            self._last_analysed_s is not None
            and now_s - self._last_analysed_s < interval_s - _EPSILON_S
        ):
            return FrameResult(record, [], analysed=False)
        followed = self._follow(frame_bgr, now_s)
        if followed is not None:
            detections, poses = followed
        else:
            detections = self._detector.detect(frame_bgr)
            poses = (
                self._pose.estimate(frame_bgr, [d.box_xyxy for d in detections])
                if detections
                else []
            )
            self._last_detector_s = now_s
            self._followed = _to_follow(detections, poses)
        tracks = self._tracker.update(detections, poses, now_s)
        if detections:
            self._last_person_s = now_s
        self._last_analysed_s = now_s
        self._rate.record(now_s)
        track_by_index = {t.detection_index: t.track_id for t in tracks}
        people = [
            DetectedPerson(d.box_xyxy, d.score, poses[i] if poses else None, track_by_index.get(i))
            for i, d in enumerate(detections)
        ]
        return FrameResult(record, tracks, analysed=True, detections=people)

    def _follow(
        self, frame_bgr: NDArray[np.uint8], now_s: float
    ) -> tuple[list[Detection], list[Pose | None]] | None:
        """This frame's people without the detector, or ``None`` when it must run."""
        if (
            not self._followed
            or self._last_detector_s is None
            or now_s - self._last_detector_s >= self._detector_interval_s - _EPSILON_S
        ):
            return None
        height, width = frame_bgr.shape[:2]
        poses = self._pose.estimate(
            frame_bgr, [search_box(f.box_xyxy, width=width, height=height) for f in self._followed]
        )
        detections: list[Detection] = []
        carried: list[Followed] = []
        for person, pose in zip(self._followed, poses, strict=True):
            kp_box = keypoint_box(pose) if pose is not None else None
            if kp_box is None:
                return None
            box = clip_box(apply_margins(kp_box, person.margins), width=width, height=height)
            if box_iou(box, person.box_xyxy) < MIN_FOLLOW_IOU:
                return None
            detections.append(Detection(box, person.score))
            carried.append(Followed(box, person.score, person.margins))
        if any(box_iou(a.box_xyxy, b.box_xyxy) > 0 for a, b in combinations(carried, 2)):
            return None
        self._followed = carried
        return detections, list(poses)

    def sharing_models(self, tracker: PersonTracker) -> "PerceptionPipeline":
        """A pipeline with the same loaded models and its own tracker that analyses every
        frame (full-rate confirmation): no second copy of the models in memory."""
        fresh = PerceptionPipeline(
            self._detector,
            self._pose,
            tracker,
            CadencePolicy(),
            RateMonitor(min_fps=_CONFIRMATION_MIN_FPS, started_s=0.0),
        )
        fresh.request_attention(float("inf"))
        return fresh

    def degraded(self, now_s: float) -> bool:
        return self._rate.degraded(now_s)

    def resume(self, now_s: float) -> None:
        """The camera's stream (re)connected (see ``RateMonitor.resume``)."""
        self._rate.resume(now_s)


def _to_follow(detections: Sequence[Detection], poses: Sequence[Pose | None]) -> list[Followed]:
    """The detector's people to follow; none (the detector runs next time) when a person
    has no usable pose or two people overlap."""
    followed = []
    for detection, pose in zip(detections, poses, strict=True):
        kp_box = keypoint_box(pose) if pose is not None else None
        if kp_box is None:
            return []
        box = detection.box_xyxy
        followed.append(Followed(box, detection.score, box_margins(box, kp_box)))
    if any(box_iou(a.box_xyxy, b.box_xyxy) > 0 for a, b in combinations(followed, 2)):
        return []
    return followed
