"""Pose windows for temporal classifiers (#15): observed frames only, one reference."""

import numpy as np
import pytest

from fall_impact_monitor.analysis.pose_window import (
    FEATURES,
    WINDOW_OBSERVATIONS,
    PoseObservation,
    PoseWindows,
    window_features,
)
from fall_impact_monitor.api.perception import FrameResult
from fall_impact_monitor.perception.tracking import Track
from tests.api.test_facade import record

L_HIP, R_HIP = 11, 12


def person(t_s: float, x: float = 100.0, y: float = 50.0, height: float = 100.0) -> PoseObservation:
    """Keypoints spread down a box of ``height`` whose top-left corner is (x, y)."""
    keypoints = np.stack([np.full(17, x + 10.0), y + np.linspace(0.0, height, 17)], axis=1).astype(
        np.float32
    )
    return PoseObservation(
        t_s, (x, y, x + 20.0, y + height), keypoints, np.full(17, 0.9, np.float32)
    )


def test_a_short_track_is_padded_before_its_observed_rows() -> None:
    features = window_features([person(0.0), person(0.125), person(0.3)])
    assert features.shape == (WINDOW_OBSERVATIONS, FEATURES)
    observed = features[:, -1]
    assert observed.tolist() == [0.0] * (WINDOW_OBSERVATIONS - 3) + [1.0] * 3
    assert not features[: WINDOW_OBSERVATIONS - 3].any()  # padding is all zeros


def test_time_gaps_are_the_real_ones() -> None:
    features = window_features([person(0.0), person(0.125), person(0.5)])
    assert features[-3:, -2].tolist() == pytest.approx([0.0, 0.125, 0.375])


def test_only_the_latest_observations_are_kept() -> None:
    observations = [person(i * 0.125) for i in range(40)]
    features = window_features(observations)
    assert features[:, -1].sum() == WINDOW_OBSERVATIONS
    assert features[-1, -2] == pytest.approx(0.125)


def test_position_and_size_in_the_image_do_not_matter() -> None:
    near = [person(t, y=50.0 + 40.0 * t) for t in (0.0, 0.125, 0.25)]
    far = [
        PoseObservation(
            o.t_s,
            (o.box[0] * 0.5 + 300, o.box[1] * 0.5 + 10, o.box[2] * 0.5 + 300, o.box[3] * 0.5 + 10),
            o.keypoints * 0.5 + np.array([300.0, 10.0], np.float32),
            o.scores,
        )
        for o in near
    ]
    assert np.allclose(window_features(near), window_features(far), atol=1e-5)


def test_a_drop_stays_visible_across_the_window() -> None:
    falling = [person(t, y=50.0 + 200.0 * t) for t in (0.0, 0.125, 0.25, 0.375)]
    features = window_features(falling)
    hip_y = (features[-4:, L_HIP * 3 + 1] + features[-4:, R_HIP * 3 + 1]) / 2
    assert np.all(np.diff(hip_y) > 0)  # image y grows downwards: the hips go down
    assert hip_y[0] == pytest.approx(0.0, abs=1e-6)  # the first hip midpoint is the reference


def test_confidences_are_kept_for_every_keypoint() -> None:
    o = person(0.0)
    o.scores[3] = 0.1
    features = window_features([o])
    assert features[-1, 3 * 3 + 2] == pytest.approx(0.1)
    assert features[-1, 0 * 3 + 2] == pytest.approx(0.9)


def make_track(track_id: int, o: PoseObservation) -> Track:
    return Track(track_id, o.box, 0.9, o.keypoints, o.scores, "stable", 10, 0)


def test_windows_follow_each_track_with_keypoints() -> None:
    windows = PoseWindows()
    for i in range(5):
        t = i * 0.125
        tracks = [make_track(1, person(t)), make_track(2, person(t, x=300.0))]
        tracks.append(Track(3, (0.0, 0.0, 10.0, 10.0), 0.5, None, None, "uncertain", 1, 2))
        ready = windows.update(FrameResult(record(t, i), tracks, analysed=True))
    assert sorted(track_id for track_id, _ in ready) == [1, 2]  # no keypoints, no window
    assert all(features[:, -1].sum() == 5 for _, features in ready)


def test_a_track_seen_again_after_a_long_gap_starts_a_new_window() -> None:
    windows = PoseWindows(max_gap_s=1.0)
    for i, t in enumerate((0.0, 0.125, 0.25, 3.0)):
        ready = dict(
            windows.update(FrameResult(record(t, i), [make_track(1, person(t))], analysed=True))
        )
    assert ready[1][:, -1].sum() == 1
