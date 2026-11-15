"""Labelled pose windows from saved analyses, for temporal classifiers (#15)."""

import json
from pathlib import Path

import numpy as np
import pytest

from fall_impact_monitor.analysis.pose_window import FEATURES, WINDOW_OBSERVATIONS
from fall_impact_monitor.cli.main import main
from fall_impact_monitor.evaluation.truth import TruthEvent
from fall_impact_monitor.evaluation.windows import IGNORE_AFTER_S, group_of, window_label
from tests.cli.test_eval_cli import gmdcsa24
from tests.cli.test_replay import scripted  # noqa: F401  (fixture)

FALL = (TruthEvent(10.0, 11.0),)


@pytest.mark.parametrize(
    ("start_s", "end_s", "label"),
    [
        (8.0, 10.0, 1),  # ends as the fall starts
        (10.2, 10.8, 1),  # inside the fall
        (11.0, 13.0, 1),  # starts as it ends
        (6.0, 9.9, 0),  # before
        (11.1, 11.0 + IGNORE_AFTER_S, None),  # lying right after: neither
        (11.5, 11.0 + IGNORE_AFTER_S + 0.1, 0),  # later
    ],
)
def test_a_window_is_a_fall_when_it_overlaps_one(
    start_s: float, end_s: float, label: int | None
) -> None:
    assert window_label(start_s, end_s, FALL) == label


@pytest.mark.parametrize(
    ("dataset", "name", "group"),
    [
        ("gmdcsa24", "Subject 3/Fall/01.mp4", "gmdcsa24:Subject 3"),
        ("caucafall", "CAUCAFall/Subject.7/Kneel/KneelS7.avi", "caucafall:Subject.7"),
        ("urfd", "raw/videos/fall-08-cam0.mp4", "urfd:fall-08"),
        ("mcfd", "dataset/chute05/cam3.avi", "mcfd:chute05"),
    ],
)
def test_groups_keep_a_person_or_an_event_on_one_side_of_a_split(
    dataset: str, name: str, group: str
) -> None:
    assert group_of(dataset, name) == group


def test_windows_are_exported_from_saved_analyses(tmp_path: Path, scripted: None) -> None:  # noqa: F811
    data, out = tmp_path / "data", tmp_path / "report"
    gmdcsa24(data)
    eval_args = ["eval", "--dataset", "gmdcsa24", "--root", str(data), "--out-dir", str(out)]
    assert main(eval_args) == 0
    output = tmp_path / "windows.npz"
    args = ["windows", "--dataset", "gmdcsa24", "--root", str(data), "--out-dir", str(out)]
    assert main([*args, "--output", str(output)]) == 0
    with np.load(output, allow_pickle=False) as saved:
        x, y, groups = saved["features"], saved["labels"], saved["groups"]
        meta = json.loads(str(saved["meta"]))
    assert x.shape[1:] == (WINDOW_OBSERVATIONS, FEATURES) and x.dtype == np.float32
    assert len(x) == len(y) == len(groups) > 0
    assert set(np.unique(y)) <= {0, 1} and 1 in y and 0 in y
    assert set(groups) == {"gmdcsa24:Subject 1"}
    assert meta["perception"]["FIM_ANALYSIS_PERSON_VISIBLE_FPS"] == 8.0


def test_windows_need_complete_analyses(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    gmdcsa24(tmp_path / "data")
    args = ["windows", "--dataset", "gmdcsa24", "--root", str(tmp_path / "data")]
    assert main([*args, "--out-dir", str(tmp_path / "none"), "--output", "x.npz"]) == 2
    assert "fim eval" in capsys.readouterr().out


def test_padded_frames_never_become_training_windows() -> None:
    """``fim eval`` pads clip ends with the last frame repeated (#81): a frozen image is
    evaluation scaffolding, not something a classifier should learn from."""
    from fall_impact_monitor.cli.analyze import frame_entry
    from fall_impact_monitor.evaluation.windows import labelled_windows
    from tests.analysis.test_fall_analyzer import frame, standing

    real = [frame_entry(frame(seq, [standing()]), (640, 480)) for seq in range(60)]
    padded = [
        {**frame_entry(frame(seq, [standing()]), (640, 480)), "padded": True}
        for seq in range(60, 120)
    ]
    only_real = labelled_windows(iter(real), FALL, 0.0)
    assert only_real
    assert len(labelled_windows(iter(real + padded), FALL, 0.0)) == len(only_real)
