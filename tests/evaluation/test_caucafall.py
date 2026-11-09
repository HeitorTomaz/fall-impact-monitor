from pathlib import Path

import pytest

from fall_impact_monitor.evaluation.caucafall import LabeledBox, load_caucafall_labels, match_recall


def test_labels_are_parsed_to_zero_based_frames_and_pixels(tmp_path: Path) -> None:
    video = tmp_path / "FallForwardS10.avi"
    video.write_bytes(b"")
    (tmp_path / "cfs1000001.txt").write_text("0 0.5 0.5 0.2 0.4\n")
    (tmp_path / "cfs1000002.txt").write_text("1 0.25 0.75 0.5 0.5\n")
    (tmp_path / "classes.txt").write_text("nofall\nfall\n")
    labels = load_caucafall_labels(video, width=200, height=100)
    assert labels[0] == [LabeledBox((80.0, 30.0, 120.0, 70.0), fall=False)]
    assert labels[1] == [LabeledBox((0.0, 50.0, 100.0, 100.0), fall=True)]


def test_empty_label_file_means_no_person(tmp_path: Path) -> None:
    video = tmp_path / "WalkS1.avi"
    video.write_bytes(b"")
    (tmp_path / "cw100001.txt").write_text("")
    assert load_caucafall_labels(video, width=10, height=10) == {0: []}


def test_match_recall_counts_ground_truth_found_at_iou_threshold() -> None:
    truth = [(0.0, 0.0, 10.0, 10.0), (100.0, 100.0, 110.0, 110.0)]
    detections = [(1.0, 1.0, 11.0, 11.0), (50.0, 50.0, 60.0, 60.0)]
    assert match_recall(truth, detections, iou=0.5) == (1, 2)
    assert match_recall([], detections, iou=0.5) == (0, 0)


def test_one_detection_cannot_match_two_ground_truth_boxes() -> None:
    truth = [(0.0, 0.0, 10.0, 10.0), (0.5, 0.5, 10.5, 10.5)]
    assert match_recall(truth, [(0.0, 0.0, 10.0, 10.0)], iou=0.5) == (1, 2)


def test_missing_label_directory_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_caucafall_labels(tmp_path / "missing" / "x.avi", width=1, height=1)


def test_select_frames_balances_fall_and_non_fall_frames() -> None:
    from fall_impact_monitor.evaluation.caucafall import select_frames

    labels = {i: [LabeledBox((0, 0, 1, 1), fall=i >= 60)] for i in range(100)}
    labels[100] = []
    chosen = select_frames(labels, 8)
    assert len(chosen) == 8
    assert sum(1 for f in chosen if f >= 60) == 4
    assert chosen == sorted(chosen)
