"""Perception pipeline: frames in, tracked people (box, pose, identity) out."""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from fall_impact_monitor.capture.timing import FrameRecord
from fall_impact_monitor.perception.cadence import CadencePolicy, RateMonitor, SceneState
from fall_impact_monitor.perception.detector import Detection
from fall_impact_monitor.perception.pose import Box, Pose
from fall_impact_monitor.perception.tracking import PersonTracker, Track

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
        detections = self._detector.detect(frame_bgr)
        poses = (
            self._pose.estimate(frame_bgr, [d.box_xyxy for d in detections]) if detections else []
        )
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

    def degraded(self, now_s: float) -> bool:
        return self._rate.degraded(now_s)
