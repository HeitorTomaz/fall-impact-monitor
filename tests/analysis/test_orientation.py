import numpy as np

from fall_impact_monitor.analysis.orientation import OrientationCheck, upright_vote
from fall_impact_monitor.perception.tracking import Track


def person(shoulder_y: float, hip_y: float, *, dx: float = 0.0, score: float = 0.9) -> Track:
    keypoints = np.zeros((17, 2), np.float32)
    scores = np.full(17, score, np.float32)
    keypoints[5] = keypoints[6] = (100.0, shoulder_y)  # shoulders
    keypoints[11] = keypoints[12] = (100.0 + dx, hip_y)  # hips
    return Track(1, (50.0, 0.0, 150.0, 300.0), 0.9, keypoints, scores, "stable", 10, 0)


def test_an_upright_person_votes_by_where_the_shoulders_are() -> None:
    assert upright_vote(person(100.0, 200.0)) is False  # shoulders above the hips
    assert upright_vote(person(200.0, 100.0)) is True  # shoulders below: image upside down


def test_lying_or_unsure_people_do_not_vote() -> None:
    assert upright_vote(person(100.0, 120.0, dx=100.0)) is None  # torso mostly horizontal
    assert upright_vote(person(100.0, 200.0, score=0.1)) is None  # joints not visible
    no_pose = Track(1, (0.0, 0.0, 1.0, 1.0), 0.9, None, None, "stable", 10, 0)
    assert upright_vote(no_pose) is None


def test_most_upright_people_upside_down_flags_the_camera_once() -> None:
    check = OrientationCheck(min_votes=10, inverted_ratio=0.8)
    flagged = [check.update([person(200.0, 100.0)]) for _ in range(9)]
    assert flagged == [False] * 9 and not check.inverted
    assert check.update([person(200.0, 100.0)]) is True  # the tenth vote decides
    assert check.inverted
    assert check.update([person(200.0, 100.0)]) is False  # reported once


def test_a_normal_camera_is_not_flagged() -> None:
    check = OrientationCheck(min_votes=10, inverted_ratio=0.8)
    for i in range(30):
        # a few odd poses (bending over, a child upside down on the sofa) do not flag it
        check.update([person(200.0, 100.0) if i % 5 == 0 else person(100.0, 200.0)])
    assert not check.inverted


def test_a_camera_flipped_after_a_long_upright_period_is_flagged_and_cleared() -> None:
    # A week upright, then a reset turns the camera's own image flip off.
    check = OrientationCheck(min_votes=10, inverted_ratio=0.8, window_votes=50)
    for _ in range(10_000):
        check.update([person(100.0, 200.0)])
    for _ in range(50):
        check.update([person(200.0, 100.0)])
    assert check.inverted
    for _ in range(50):  # fixed again: the warning goes away
        check.update([person(100.0, 200.0)])
    assert not check.inverted
