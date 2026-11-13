import numpy as np
import pytest

from fall_impact_monitor.perception.follow import (
    apply_margins,
    box_margins,
    keypoint_box,
    search_box,
    touches_edge,
)
from fall_impact_monitor.perception.pose import Pose


def pose_with(points: list[tuple[float, float]], score: float = 0.9) -> Pose:
    keypoints = np.zeros((17, 2), np.float32)
    scores = np.zeros(17, np.float32)
    for i, point in enumerate(points):
        keypoints[i] = point
        scores[i] = score
    return Pose(keypoints, scores)


SPREAD = [(20.0, 10.0), (40.0, 10.0), (20.0, 90.0), (40.0, 90.0), (30.0, 50.0), (25.0, 60.0)]


def test_keypoint_box_spans_the_confident_keypoints() -> None:
    assert keypoint_box(pose_with(SPREAD)) == (20.0, 10.0, 40.0, 90.0)


def test_keypoint_box_ignores_unconfident_keypoints() -> None:
    pose = pose_with(SPREAD)
    pose.keypoints[10] = (500.0, 500.0)
    pose.scores[10] = 0.1
    assert keypoint_box(pose) == (20.0, 10.0, 40.0, 90.0)


def test_too_few_confident_keypoints_is_a_lost_pose() -> None:
    assert keypoint_box(pose_with(SPREAD[:3])) is None
    assert keypoint_box(pose_with(SPREAD, score=0.1)) is None


def test_margins_round_trip_to_the_detector_box() -> None:
    detector_box = (15.0, 2.0, 46.0, 100.0)
    kp_box = (20.0, 10.0, 40.0, 90.0)
    assert apply_margins(kp_box, box_margins(detector_box, kp_box)) == pytest.approx(detector_box)


def test_margins_scale_with_the_keypoint_box() -> None:
    margins = box_margins((15.0, 2.0, 46.0, 100.0), (20.0, 10.0, 40.0, 90.0))
    # the same person twice as large: margins twice as large
    assert apply_margins((40.0, 20.0, 80.0, 180.0), margins) == pytest.approx(
        (30.0, 4.0, 92.0, 200.0)
    )


def test_search_box_is_wider_and_stays_inside_the_image() -> None:
    assert search_box((40.0, 20.0, 80.0, 100.0), width=160, height=120) == pytest.approx(
        (30.0, 0.0, 90.0, 119.0)
    )


def test_touches_edge() -> None:
    assert touches_edge((0.0, 10.0, 50.0, 60.0), width=160, height=120)
    assert touches_edge((10.0, 10.0, 160.0, 60.0), width=160, height=120)
    assert not touches_edge((10.0, 10.0, 150.0, 110.0), width=160, height=120)
