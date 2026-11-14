from pathlib import Path

import pytest

from fall_impact_monitor.evaluation.mcfd import mcfd_videos
from fall_impact_monitor.evaluation.truth import TruthEvent

FPS = 30.0


def tree(root: Path, *names: str) -> Path:
    for name in names:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")
    return root


def falls(root: Path) -> dict[str, tuple[TruthEvent, ...]]:
    return {v.name: v.falls for v in mcfd_videos(root, fps_of=lambda path: FPS)}


def at(first: int, last: int) -> TruthEvent:
    """Frames numbered from 1, as in the technical report: frame n spans (n-1)/fps..n/fps."""
    return TruthEvent((first - 1) / FPS, last / FPS)


def test_falls_on_the_reference_camera_include_lying_on_the_ground(tmp_path: Path) -> None:
    # Scenario 4, reference camera 6: falling 288-314 then lying 315-380; again 601-780.
    found = falls(tree(tmp_path, "dataset/chute04/cam6.avi"))
    assert found["dataset/chute04/cam6.avi"] == (at(288, 380), at(601, 780))


def test_other_cameras_are_aligned_by_their_delays(tmp_path: Path) -> None:
    # Scenario 4: delay 82 frames on camera 6 (reference), 0 on camera 4.
    found = falls(tree(tmp_path, "dataset/chute04/cam4.avi"))
    assert found["dataset/chute04/cam4.avi"] == (at(206, 298), at(519, 698))


def test_the_erratum_corrects_periods_reference_cameras_and_delays(tmp_path: Path) -> None:
    root = tree(tmp_path, "dataset/chute02/cam4.avi", "dataset/chute05/cam5.avi")
    found = falls(root)
    # Scenario 2 periods corrected to falling 395-425, lying 426-600.
    assert found["dataset/chute02/cam4.avi"] == (at(395, 600),)
    # Scenario 5: reference camera "11" is camera 8 (delay 0); camera 5's delay is 18.
    assert found["dataset/chute05/cam5.avi"] == (at(329, 468),)


def test_a_fall_with_recovery_and_a_scenario_without_falls(tmp_path: Path) -> None:
    root = tree(tmp_path, "dataset/chute23/cam8.avi", "dataset/chute24/cam6.avi")
    found = falls(root)
    assert found["dataset/chute23/cam8.avi"] == (at(3574, 3614),)  # got up at once
    assert found["dataset/chute24/cam6.avi"] == ()


def test_every_scenario_and_camera_is_found_in_order(tmp_path: Path) -> None:
    names = [f"dataset/chute{s:02d}/cam{c}.avi" for s in range(1, 25) for c in range(1, 9)]
    videos = mcfd_videos(tree(tmp_path, *names, "dataset/notes.txt"), fps_of=lambda p: FPS)
    assert [v.name for v in videos] == names
    assert sum(len(v.falls) for v in videos) == 24 * 8  # 23 scenarios, two falls in one


def test_an_unknown_scenario_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="chute25"):
        mcfd_videos(tree(tmp_path, "dataset/chute25/cam1.avi"), fps_of=lambda p: FPS)
