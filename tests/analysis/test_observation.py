import numpy as np
import pytest

from fall_impact_monitor.analysis.observation import Observation, observe

BOX = (100.0, 50.0, 140.0, 250.0)  # 40 x 200 upright


def keypoints(
    shoulder: tuple[float, float], hip: tuple[float, float]
) -> tuple[np.ndarray, np.ndarray]:
    points = np.zeros((17, 2), np.float32)
    scores = np.full(17, 0.9, np.float32)
    points[5], points[6] = (shoulder[0] - 10, shoulder[1]), (shoulder[0] + 10, shoulder[1])
    points[11], points[12] = (hip[0] - 8, hip[1]), (hip[0] + 8, hip[1])
    return points, scores


def test_upright_person() -> None:
    points, scores = keypoints((120, 90), (120, 150))
    obs = observe(BOX, points, scores, 0.9, 1.0)
    assert isinstance(obs, Observation) and obs.valid
    assert obs.hip_y == pytest.approx(150.0)
    assert obs.torso_angle_deg == pytest.approx(0.0)
    assert obs.aspect == pytest.approx(0.2)
    assert obs.height_px == pytest.approx(200.0)


def test_lying_person_has_horizontal_torso() -> None:
    points, scores = keypoints((100, 200), (160, 200))
    obs = observe((60.0, 180.0, 260.0, 230.0), points, scores, 0.8, 2.0)
    assert obs.torso_angle_deg == pytest.approx(90.0)
    assert obs.aspect == pytest.approx(4.0)


def test_one_visible_side_is_enough() -> None:
    points, scores = keypoints((120, 90), (120, 150))
    scores[[6, 12]] = 0.0
    assert observe(BOX, points, scores, 0.9, 1.0).valid


def test_missing_hips_or_shoulders_is_invalid_but_keeps_box_features() -> None:
    points, scores = keypoints((120, 90), (120, 150))
    scores[[11, 12]] = 0.0
    obs = observe(BOX, points, scores, 0.9, 1.0)
    assert not obs.valid
    assert obs.aspect == pytest.approx(0.2)


def test_no_pose_is_invalid() -> None:
    obs = observe(BOX, None, None, 0.5, 1.0)
    assert not obs.valid and obs.t_s == 1.0
