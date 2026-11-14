"""Real falls of older adults: the SFU public videos with local annotations (#72)."""

from pathlib import Path

import pytest

from fall_impact_monitor.evaluation.sfu import sfu_public_videos
from fall_impact_monitor.evaluation.truth import TruthEvent
from fall_impact_monitor.evaluation.windows import group_of

HEADER = "clip,participant,fall_start_s,fall_end_s,quality,notes\n"


def dataset(root: Path, rows: str, clips: list[str]) -> Path:
    (root / "clips").mkdir(parents=True)
    for clip in clips:
        (root / "clips" / f"{clip}.mp4").write_bytes(b"")
    (root / "labels.csv").write_text(HEADER + rows, encoding="utf-8")
    return root


def test_visible_falls_are_read_and_invisible_ones_left_out(tmp_path: Path) -> None:
    root = dataset(
        tmp_path,
        "1001_view1,P0001,14.0,15.5,clear,a\n"
        "1001_view3,P0001,11.5,12.6,uncertain,b\n"
        "1002_view1,P0002,,,not_visible,c\n",
        ["1001_view1", "1001_view3", "1002_view1"],
    )
    videos = sfu_public_videos(root)
    assert [v.name for v in videos] == ["P0001/1001_view1.mp4", "P0001/1001_view3.mp4"]
    assert videos[0].falls == (TruthEvent(14.0, 15.5),)
    assert videos[0].path == root / "clips" / "1001_view1.mp4"


def test_the_clear_subset_keeps_only_clear_annotations(tmp_path: Path) -> None:
    root = dataset(
        tmp_path,
        "1001_view1,P0001,14.0,15.5,clear,a\n1001_view3,P0001,11.5,12.6,uncertain,b\n",
        ["1001_view1", "1001_view3"],
    )
    assert [v.name for v in sfu_public_videos(root, clear_only=True)] == ["P0001/1001_view1.mp4"]


def test_a_missing_clip_is_an_error(tmp_path: Path) -> None:
    root = dataset(tmp_path, "1001_view1,P0001,14.0,15.5,clear,a\n", [])
    with pytest.raises(FileNotFoundError, match="1001_view1"):
        sfu_public_videos(root)


def test_windows_are_grouped_by_participant() -> None:
    assert group_of("sfu_public", "P0001/1001_view1.mp4") == "sfu_public:P0001"
    assert group_of("sfu_public_clear", "P0002/1003_view2.mp4") == "sfu_public_clear:P0002"
