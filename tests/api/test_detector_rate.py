"""The person detector at its own, lower rate; pose on every analysed frame (#68)."""

from collections.abc import Callable, Sequence

import numpy as np
import pytest

from fall_impact_monitor.api import FrameResult, PerceptionPipeline
from fall_impact_monitor.perception.cadence import CadencePolicy, RateMonitor
from fall_impact_monitor.perception.detector import Detection
from fall_impact_monitor.perception.pose import Box, Pose
from fall_impact_monitor.perception.tracking import PersonTracker
from tests.api.test_facade import record

FRAME = np.zeros((240, 320, 3), np.uint8)
STANDING: Box = (140.0, 40.0, 180.0, 200.0)

Scene = Callable[[float], list[Box]]  # time -> true person boxes


def inner_keypoints(box: Box) -> np.ndarray:
    """17 keypoints spread over the box shrunk by 10 % on each side."""
    x1, y1, x2, y2 = box
    dx, dy = 0.1 * (x2 - x1), 0.1 * (y2 - y1)
    xs = np.linspace(x1 + dx, x2 - dx, 17)
    ys = np.linspace(y1 + dy, y2 - dy, 17)
    return np.stack([xs, ys], axis=1).astype(np.float32)


class SceneDetector:
    def __init__(self, scene: Scene) -> None:
        self.scene = scene
        self.now_s = 0.0
        self.calls: list[float] = []

    def detect(self, image_bgr: np.ndarray) -> list[Detection]:
        self.calls.append(self.now_s)
        return [Detection(box, 0.9) for box in self.scene(self.now_s)]


class ScenePose:
    """Finds the true person whose box overlaps the given box most."""

    def __init__(self, scene: Scene, *, lost_after_s: float = float("inf")) -> None:
        self.scene = scene
        self.now_s = 0.0
        self.lost_after_s = lost_after_s
        self.boxes: list[list[Box]] = []

    def estimate(self, image_bgr: np.ndarray, boxes: Sequence[Box]) -> list[Pose | None]:
        self.boxes.append(list(boxes))
        if self.now_s >= self.lost_after_s:
            return [None for _ in boxes]
        poses: list[Pose | None] = []
        for box in boxes:
            inside = [p for p in self.scene(self.now_s) if _overlap(p, box) > 0]
            if not inside:
                poses.append(None)
                continue
            person = max(inside, key=lambda p: _overlap(p, box))
            poses.append(Pose(inner_keypoints(person), np.full(17, 0.9, np.float32)))
        return poses


def _overlap(a: Box, b: Box) -> float:
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))


def run(
    scene: Scene, seconds: float, *, detector_max_fps: float, lost_after_s: float = float("inf")
) -> tuple[list[FrameResult], SceneDetector, ScenePose]:
    detector, pose = SceneDetector(scene), ScenePose(scene, lost_after_s=lost_after_s)
    pipe = PerceptionPipeline(
        detector,
        pose,
        PersonTracker(),
        CadencePolicy(empty_fps=1.0, person_fps=8.0),
        RateMonitor(min_fps=0.5, started_s=0.0),
        detector_max_fps=detector_max_fps,
    )
    results = []
    for i in range(int(seconds * 25)):
        t_s = i / 25
        detector.now_s = pose.now_s = t_s
        results.append(pipe.process(FRAME, record(t_s, i)))
    return results, detector, pose


def still(t_s: float) -> list[Box]:
    return [STANDING]


def test_detector_runs_at_its_own_rate_and_pose_on_every_analysed_frame() -> None:
    results, detector, pose = run(still, 4.0, detector_max_fps=2.0)
    analysed = [r for r in results if r.analysed]
    assert len(analysed) == 25  # every 4th frame of 25 fps: the 8 fps interval
    assert len(detector.calls) <= 9  # 2 per second
    assert len(pose.boxes) == len(analysed)
    assert all(len(r.detections) == 1 and r.detections[0].pose for r in analysed)


def test_followed_box_matches_the_detector_box_of_a_still_person() -> None:
    results, detector, _ = run(still, 2.0, detector_max_fps=2.0)
    followed = [
        r for r in results if r.analysed and r.record.arrival_ns / 1e9 not in detector.calls
    ]
    assert followed
    for result in followed:
        assert result.detections[0].box_xyxy == pytest.approx(STANDING, abs=1e-3)
        assert result.detections[0].score == 0.9


def test_followed_people_keep_their_track() -> None:
    results, _, _ = run(still, 3.0, detector_max_fps=2.0)
    analysed = [r for r in results if r.analysed]
    assert {t.track_id for r in analysed[-10:] for t in r.tracks} == {
        analysed[-1].tracks[0].track_id
    }


def test_zero_rate_runs_the_detector_on_every_analysed_frame() -> None:
    results, detector, _ = run(still, 2.0, detector_max_fps=0.0)
    assert len(detector.calls) == sum(r.analysed for r in results)


def test_a_lost_pose_runs_the_detector_at_once() -> None:
    results, detector, _ = run(still, 2.0, detector_max_fps=1.0, lost_after_s=1.2)
    after = [
        r.record.arrival_ns / 1e9 for r in results if r.analysed and r.record.arrival_ns >= 1.2e9
    ]
    assert after
    assert set(after) <= set(detector.calls)


def test_a_falling_person_runs_the_detector_at_once() -> None:
    lying: Box = (100.0, 160.0, 260.0, 200.0)

    def fall(t_s: float) -> list[Box]:
        return [STANDING if t_s < 1.0 else lying]

    results, detector, _ = run(fall, 2.0, detector_max_fps=1.0)
    first_after = next(r for r in results if r.analysed and r.record.arrival_ns / 1e9 >= 1.0)
    assert first_after.record.arrival_ns / 1e9 in detector.calls
    assert first_after.detections[0].box_xyxy == lying


def test_a_person_who_leaves_the_image_is_dropped_at_once() -> None:
    def walk_out(t_s: float) -> list[Box]:
        x = 140.0 + 150.0 * t_s  # gone at 1.2 s
        return [(x, 40.0, min(x + 40.0, 320.0), 200.0)] if x < 320.0 else []

    results, detector, _ = run(walk_out, 1.6, detector_max_fps=1.0)
    analysed = [r for r in results if r.analysed]
    assert all(d.box_xyxy[2] <= 320.0 for r in analysed for d in r.detections)
    gone = next(r for r in analysed if r.record.arrival_ns >= 1.2e9)
    assert gone.record.arrival_ns / 1e9 in detector.calls  # the pose was lost
    assert gone.detections == []


def test_a_person_already_at_the_edge_is_followed() -> None:
    # A camera close to people often cuts them at the border of the image.
    cut: Box = (140.0, 120.0, 180.0, 240.0)

    def stay(t_s: float) -> list[Box]:
        return [cut]

    results, detector, _ = run(stay, 4.0, detector_max_fps=2.0)
    analysed = [r for r in results if r.analysed]
    assert len(detector.calls) <= 9
    assert all(r.detections[0].box_xyxy[3] <= 240.0 for r in analysed)


def test_a_person_drifting_onto_the_border_is_followed() -> None:
    # Close to the camera, a person's feet come and go at the bottom of the image.
    def drift(t_s: float) -> list[Box]:
        y2 = 228.0 + 12.0 * (t_s % 1.0)
        return [(140.0, y2 - 150.0, 180.0, min(y2, 240.0))]

    results, detector, _ = run(drift, 4.0, detector_max_fps=2.0)
    _, still_detector, _ = run(still, 4.0, detector_max_fps=2.0)
    assert len(detector.calls) == len(still_detector.calls)
    assert all(d.box_xyxy[3] <= 240.0 for r in results for d in r.detections)


def test_people_close_together_run_the_detector() -> None:
    def pair(t_s: float) -> list[Box]:
        return [STANDING, (175.0, 40.0, 215.0, 200.0)]

    results, detector, _ = run(pair, 2.0, detector_max_fps=1.0)
    assert len(detector.calls) == sum(r.analysed for r in results)


def test_a_new_person_is_found_at_the_next_detector_frame() -> None:
    def arrives(t_s: float) -> list[Box]:
        return [STANDING] if t_s < 1.0 else [STANDING, (20.0, 40.0, 60.0, 200.0)]

    results, _, _ = run(arrives, 2.0, detector_max_fps=2.0)
    seen = next(r for r in results if r.analysed and len(r.detections) == 2)
    assert 1.0 <= seen.record.arrival_ns / 1e9 <= 1.0 + 0.5 + 0.13


def test_full_rate_reanalysis_runs_the_detector_on_every_frame() -> None:
    detector, pose = SceneDetector(still), ScenePose(still)
    live = PerceptionPipeline(
        detector,
        pose,
        PersonTracker(),
        CadencePolicy(),
        RateMonitor(min_fps=0.5, started_s=0.0),
        detector_max_fps=2.0,
    )
    fresh = live.sharing_models(PersonTracker())
    for i in range(25):
        detector.now_s = pose.now_s = i / 25
        fresh.process(FRAME, record(i / 25, i))
    assert len(detector.calls) == 25
