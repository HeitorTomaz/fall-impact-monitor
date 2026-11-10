from pathlib import Path

import pytest

from fall_impact_monitor.evaluation.truth import (
    TruthEvent,
    caucafall_videos,
    gmdcsa24_videos,
    urfd_videos,
)

GMDCSA_HEADER = "File Name,Length (seconds),Time of Recording,Attire,Description, Classes\n"


def test_gmdcsa24_fall_ranges_come_from_the_classes_column(tmp_path: Path) -> None:
    subject = tmp_path / "Subject 1"
    for folder in ("Fall", "ADL"):
        (subject / folder).mkdir(parents=True)
    for name in ("01.mp4", "02.mp4"):
        (subject / "Fall" / name).touch()
    (subject / "ADL" / "01.mp4").touch()
    (subject / "Fall.csv").write_text(
        GMDCSA_HEADER
        + "01.mp4,06, Day,Shirt, Falling from a chair, Falling (SW)[3.4 to 6]; Sitting[0 to 3.4]\n"
        + "02.mp4,03,Day,Shirt,falling forward, Fall (FW)[0 to 3]\n"
    )
    (subject / "ADL.csv").write_text(
        GMDCSA_HEADER
        + "01.mp4,08, Day, Shirt, Lying on the bed, Sitting[0 to 1]; Sleeping[2 to 8]\n"
    )
    videos = {v.name: v for v in gmdcsa24_videos(tmp_path)}
    assert videos["Subject 1/Fall/01.mp4"].falls == (TruthEvent(3.4, 6.0),)
    assert videos["Subject 1/Fall/02.mp4"].falls == (TruthEvent(0.0, 3.0),)
    assert videos["Subject 1/ADL/01.mp4"].falls == ()


def test_urfd_falls_start_at_the_first_falling_frame(tmp_path: Path) -> None:
    (tmp_path / "raw" / "labels").mkdir(parents=True)
    (tmp_path / "raw" / "videos").mkdir(parents=True)
    for name in ("fall-01-cam0.mp4", "adl-01-cam0.mp4"):
        (tmp_path / "raw" / "videos" / name).touch()
    labels = [-1, -1, -1, 0, 0, 0, 1, 1, 1, 1]
    (tmp_path / "raw" / "labels" / "urfall-cam0-falls.csv").write_text(
        "".join(f"fall-01,{i + 1},{label},0,0,0,0,0,0,0,0\n" for i, label in enumerate(labels))
    )
    (tmp_path / "raw" / "labels" / "urfall-cam0-adls.csv").write_text(
        "".join(f"adl-01,{i + 1},{label},0,0,0,0,0,0,0,0\n" for i, label in enumerate(labels))
    )
    videos = {v.name: v for v in urfd_videos(tmp_path, fps=30.0)}
    (fall,) = videos["raw/videos/fall-01-cam0.mp4"].falls
    assert fall.start_s == pytest.approx(3 / 30) and fall.end_s == pytest.approx(10 / 30)
    assert videos["raw/videos/adl-01-cam0.mp4"].falls == ()  # lying down on purpose is not a fall


def test_caucafall_falls_start_at_the_first_fall_labelled_frame(tmp_path: Path) -> None:
    fall = tmp_path / "CAUCAFall" / "Subject.1" / "Fall forward"
    walk = tmp_path / "CAUCAFall" / "Subject.1" / "Walk"
    for folder, video in ((fall, "FallForwardS1.avi"), (walk, "WalkS1.avi")):
        folder.mkdir(parents=True)
        (folder / video).touch()
        (folder / "classes.txt").write_text("nofall\nfall\n")
    for i in range(1, 11):
        (fall / f"cff1{i:05d}.txt").write_text(f"{1 if i > 5 else 0} 0.5 0.5 0.2 0.4\n")
        (walk / f"cws1{i:05d}.txt").write_text("0 0.5 0.5 0.2 0.4\n")
    videos = {v.name: v for v in caucafall_videos(tmp_path, fps_of=lambda path: 10.0)}
    (event,) = videos["CAUCAFall/Subject.1/Fall forward/FallForwardS1.avi"].falls
    assert event.start_s == pytest.approx(0.5) and event.end_s == pytest.approx(1.0)
    assert videos["CAUCAFall/Subject.1/Walk/WalkS1.avi"].falls == ()


def test_a_gmdcsa24_fall_without_a_range_lasts_from_the_last_range_to_the_clip_end(
    tmp_path: Path,
) -> None:
    subject = tmp_path / "Subject 4"
    (subject / "Fall").mkdir(parents=True)
    (subject / "Fall" / "15.mp4").touch()
    (subject / "Fall.csv").write_text(
        GMDCSA_HEADER
        + "15.mp4,04,Day,Shirt,Falling from the bed, Falling (FW); Sitting[0 to 1.7]\n"
    )
    (video,) = gmdcsa24_videos(tmp_path)
    assert video.falls == (TruthEvent(1.7, 4.0),)


def gmdcsa_row(tmp_path: Path, classes: str, length: str = "10") -> list[TruthEvent]:
    subject = tmp_path / "Subject 1"
    (subject / "Fall").mkdir(parents=True, exist_ok=True)
    (subject / "Fall" / "09.mp4").touch()
    (subject / "Fall.csv").write_text(
        GMDCSA_HEADER + f"09.mp4,{length},Day,Shirt,a fall, {classes}\n"
    )
    (video,) = gmdcsa24_videos(tmp_path)
    return list(video.falls)


def test_a_gmdcsa24_range_without_to_is_read(tmp_path: Path) -> None:
    assert gmdcsa_row(tmp_path, "Walking[0 to 2]; Falling (BW)[2.5 4]") == [TruthEvent(2.5, 4.0)]


def test_falls_that_touch_are_one_fall(tmp_path: Path) -> None:
    falls = gmdcsa_row(tmp_path, "Falling (BW)[2 to 4]; Falling (SW)[4 to 11]", "11")
    assert falls == [TruthEvent(2.0, 11.0)]


def test_a_fall_row_without_any_fall_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"09.mp4"):
        gmdcsa_row(tmp_path, "Sitting[0 to 3]")
