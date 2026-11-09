from pathlib import Path
from typing import Any

import numpy as np
import pytest

from fall_impact_monitor.perception.pose import (
    PoseEstimator,
    crop_geometry,
    decode_keypoints,
)

MODELS = Path(__file__).parents[2] / "models"


def test_wide_lying_box_is_expanded_not_distorted() -> None:
    center, scale = crop_geometry((0.0, 0.0, 200.0, 50.0))
    assert center.tolist() == [100.0, 25.0]
    assert scale[0] == pytest.approx(250.0)  # 200 * 1.25
    assert scale[0] / scale[1] == pytest.approx(192 / 256)  # height grows to keep 0.75


def test_tall_box_keeps_height_and_widens() -> None:
    _, scale = crop_geometry((0.0, 0.0, 50.0, 400.0))
    assert scale[1] == pytest.approx(500.0)
    assert scale[0] == pytest.approx(375.0)


def simcc_with_peaks(x_index: int, y_index: int, value: float = 0.9) -> tuple[np.ndarray, ...]:
    simcc_x = np.zeros((1, 17, 384), np.float32)
    simcc_y = np.zeros((1, 17, 512), np.float32)
    simcc_x[0, :, x_index] = value
    simcc_y[0, :, y_index] = value
    return simcc_x, simcc_y


def test_decode_keypoints_maps_crop_to_image_coordinates() -> None:
    simcc_x, simcc_y = simcc_with_peaks(100, 200)
    keypoints, scores = decode_keypoints(
        simcc_x[0], simcc_y[0], np.array([96.0, 128.0]), np.array([192.0, 256.0])
    )
    assert keypoints.shape == (17, 2)
    assert keypoints[0].tolist() == pytest.approx([50.0, 100.0])
    assert scores[0] == pytest.approx(0.9)


class FakeInput:
    name = "input"
    shape = ("batch", 3, 256, 192)


class FakeSession:
    def __init__(self) -> None:
        self.calls = 0

    def get_inputs(self) -> list[FakeInput]:
        return [FakeInput()]

    def run(self, outputs: Any, feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
        self.calls += 1
        batch = feeds["input"].shape[0]
        assert feeds["input"].shape[1:] == (3, 256, 192)
        simcc_x, simcc_y = simcc_with_peaks(192, 256)  # crop centre
        return [np.repeat(simcc_x, batch, 0), np.repeat(simcc_y, batch, 0)]


def test_estimate_returns_one_pose_per_box_and_skips_degenerate_boxes() -> None:
    session = FakeSession()
    estimator = PoseEstimator(session)  # type: ignore[arg-type]
    image = np.zeros((300, 300, 3), np.uint8)
    poses = estimator.estimate(image, [(100.0, 50.0, 140.0, 250.0), (10.0, 10.0, 10.5, 40.0)])
    assert poses[1] is None
    pose = poses[0]
    assert pose is not None
    assert pose.keypoints[0].tolist() == pytest.approx([120.0, 150.0])  # box centre
    assert session.calls == 1


def test_no_boxes_does_not_run_the_model() -> None:
    session = FakeSession()
    assert PoseEstimator(session).estimate(np.zeros((10, 10, 3), np.uint8), []) == []  # type: ignore[arg-type]
    assert session.calls == 0


@pytest.mark.skipif(not (MODELS / "rtmpose_s.onnx").exists(), reason="model weights not available")
def test_real_model_runs(synthetic_person_frame: np.ndarray) -> None:
    estimator = PoseEstimator.from_path(MODELS / "rtmpose_s.onnx")
    (pose,) = estimator.estimate(synthetic_person_frame, [(200.0, 40.0, 320.0, 340.0)])
    assert pose is not None and pose.keypoints.shape == (17, 2) and pose.scores.shape == (17,)


def test_rotation_helpers_round_trip() -> None:
    from fall_impact_monitor.perception.pose import rotate_box, unrotate_points

    height, width = 300, 500
    box = (100.0, 200.0, 300.0, 250.0)  # wide: lying person
    point = np.array([[150.0, 220.0]], np.float32)
    for clockwise in (True, False):
        rotated = rotate_box(box, height, width, clockwise=clockwise)
        assert rotated[2] - rotated[0] == pytest.approx(50.0)  # now tall
        if clockwise:
            moved = np.array([[height - 1 - point[0, 1], point[0, 0]]], np.float32)
        else:
            moved = np.array([[point[0, 1], width - 1 - point[0, 0]]], np.float32)
        back = unrotate_points(moved, height, width, clockwise=clockwise)
        assert back[0].tolist() == pytest.approx(point[0].tolist())


def test_wide_box_runs_both_rotations_and_maps_back_to_the_box_centre() -> None:
    session = FakeSession()
    batches: list[int] = []
    original_run = session.run

    def run(outputs: Any, feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
        batches.append(feeds["input"].shape[0])
        return original_run(outputs, feeds)

    session.run = run  # type: ignore[method-assign]
    estimator = PoseEstimator(session)  # type: ignore[arg-type]
    image = np.zeros((300, 500, 3), np.uint8)
    (pose,) = estimator.estimate(image, [(100.0, 200.0, 300.0, 250.0)])
    assert batches == [2]  # clockwise and counter-clockwise crops
    assert pose is not None
    assert pose.keypoints[0].tolist() == pytest.approx([199.5, 224.5], abs=1.0)
