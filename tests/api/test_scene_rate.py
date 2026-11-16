"""Person analysis rate held by strong detections and sudden track changes, weak
detections continuing tracks at the empty-scene rate (#77)."""

from collections.abc import Callable, Sequence

import numpy as np

from fall_impact_monitor.api import FrameResult, PerceptionPipeline
from fall_impact_monitor.perception.cadence import CadencePolicy, RateMonitor
from fall_impact_monitor.perception.detector import Detection
from fall_impact_monitor.perception.pose import Box, Pose
from fall_impact_monitor.perception.tracking import PersonTracker
from tests.api.test_detector_rate import inner_keypoints
from tests.api.test_facade import record

FRAME = np.zeros((240, 320, 3), np.uint8)
STANDING: Box = (140.0, 40.0, 180.0, 200.0)
LYING: Box = (90.0, 160.0, 250.0, 200.0)
CHAIR: Box = (20.0, 150.0, 70.0, 200.0)
FPS = 25

Scene = Callable[[float], list[tuple[Box, float]]]  # time -> (box, detector score)


class ScoredDetector:
    def __init__(self, scene: Scene) -> None:
        self.scene, self.now_s = scene, 0.0

    def detect(self, image_bgr: np.ndarray) -> list[Detection]:
        return [Detection(box, score) for box, score in self.scene(self.now_s)]


class BoxPose:
    """A pose spread over whichever scene box overlaps the given box."""

    def __init__(self, scene: Scene) -> None:
        self.scene, self.now_s = scene, 0.0

    def estimate(self, image_bgr: np.ndarray, boxes: Sequence[Box]) -> list[Pose | None]:
        out: list[Pose | None] = []
        for box in boxes:
            inside = [b for b, _ in self.scene(self.now_s) if _iou_area(b, box) > 0]
            out.append(
                Pose(inner_keypoints(inside[0]), np.full(17, 0.9, np.float32)) if inside else None
            )
        return out


def _iou_area(a: Box, b: Box) -> float:
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))


def run(scene: Scene, seconds: float) -> tuple[list[FrameResult], PerceptionPipeline]:
    detector, pose = ScoredDetector(scene), BoxPose(scene)
    pipe = PerceptionPipeline(
        detector,
        pose,
        PersonTracker(new_track_min_score=0.3),
        CadencePolicy(empty_fps=3.0, person_fps=8.0),
        RateMonitor(min_fps=0.5, started_s=0.0),
        person_hold_s=10.0,
        person_min_score=0.3,
    )
    results = []
    for i in range(int(seconds * FPS)):
        t_s = i / FPS
        detector.now_s = pose.now_s = t_s
        results.append(pipe.process(FRAME, record(t_s, i)))
    return results, pipe


def analysed_between(results: list[FrameResult], a_s: float, b_s: float) -> list[FrameResult]:
    return [r for r in results if r.analysed and a_s <= r.record.arrival_ns / 1e9 < b_s]


def strong_then_weak(t_s: float) -> list[tuple[Box, float]]:
    return [(STANDING, 0.9 if t_s < 1.0 else 0.2)]


def test_strong_detection_holds_the_person_rate_for_ten_seconds() -> None:
    _, pipe = run(lambda t: [(STANDING, 0.9)] if t < 0.1 else [], 0.2)
    assert pipe.state(9.9) == "person"
    assert pipe.state(10.2) == "empty"


def test_weak_detections_alone_drop_to_the_empty_rate_after_the_hold() -> None:
    results, pipe = run(strong_then_weak, 12.0)
    assert pipe.state(11.9) == "empty"
    assert len(analysed_between(results, 2.0, 4.0)) >= 11  # every 4th frame of 25 per second


def test_the_track_is_kept_at_the_empty_rate() -> None:
    results, _ = run(strong_then_weak, 14.0)
    first = next(r for r in results if r.tracks)
    late = analysed_between(results, 12.0, 14.0)
    assert 5 <= len(late) <= 7  # 3 per second
    assert all(r.tracks and r.tracks[0].track_id == first.tracks[0].track_id for r in late)


def test_a_sudden_change_restores_the_person_rate_at_once() -> None:
    def falls(t_s: float) -> list[tuple[Box, float]]:
        return [(STANDING if t_s < 13.0 else LYING, 0.9 if t_s < 1.0 else 0.2)]

    results, pipe = run(falls, 15.5)
    # The lying box is beyond association: the person is followed again under a new id
    # once the new track is confirmed (three analysed frames at the empty rate, about 1 s),
    # and that continuation, far from where they were lost, restarts the person rate.
    assert pipe.state(15.5) == "person"
    assert analysed_between(results, 14.0, 15.5)[0].tracks
    assert len(analysed_between(results, 14.5, 15.5)) >= 5  # back to the person rate


def test_a_weak_detection_without_a_track_never_holds_the_person_rate() -> None:
    results, pipe = run(lambda t: [(CHAIR, 0.2)], 5.0)
    assert pipe.state(5.0) == "empty"
    assert all(not r.tracks for r in results)
    assert len(analysed_between(results, 1.0, 4.0)) <= 10


def test_an_unconfirmed_strong_detection_holds_the_person_rate() -> None:
    _, pipe = run(lambda t: [(STANDING, 0.9)] if t < 0.05 else [], 0.1)
    assert pipe.state(1.0) == "person"


def test_furniture_that_once_scored_strong_does_not_keep_the_person_rate() -> None:
    """A chair detected once at 0.32 opens a track; its weak detections flicker in place
    (the track goes and comes back with the same box): no sudden change, so the person
    rate ends with the hold."""

    def chair(t_s: float) -> list[tuple[Box, float]]:
        if t_s < 0.2:
            return [(CHAIR, 0.32)]
        flicker = int(t_s * 3) % 4 == 0  # missing one analysed frame in four
        return [] if flicker else [(CHAIR, 0.15)]

    results, pipe = run(chair, 24.0)
    assert pipe.state(24.0) == "empty"
    assert len(analysed_between(results, 20.0, 24.0)) <= 13  # 3 per second
