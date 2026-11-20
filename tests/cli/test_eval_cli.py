import json
from pathlib import Path

import pytest

from fall_impact_monitor.analysis.zones import load_zones
from fall_impact_monitor.cli import evaluate
from fall_impact_monitor.cli.evaluate import PAD_END_S
from fall_impact_monitor.cli.main import main
from tests.conftest import write_synthetic_video

HEADER = "File Name,Length (seconds),Time of Recording,Attire,Description, Classes\n"


def gmdcsa24(root: Path) -> None:
    subject = root / "Subject 1"
    for folder in ("Fall", "ADL"):
        (subject / folder).mkdir(parents=True)
        write_synthetic_video(subject / folder / "01.mp4")  # 3 s at 25 fps
    (subject / "Fall.csv").write_text(HEADER + "01.mp4,03,Day,Shirt,forward, Fall (FW)[1 to 2]\n")
    (subject / "ADL.csv").write_text(HEADER + "01.mp4,03,Day,Shirt,walking, Walking[0 to 3]\n")


def test_eval_scores_the_alerts_of_every_video(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    gmdcsa24(tmp_path / "data")
    alerts = {
        "Fall": [
            {"category": "possible_fall", "near_miss": False, "start_s": 1.2, "emitted_s": 2.5}
        ],
        "ADL": [
            {"category": "possible_fall", "near_miss": False, "start_s": 0.5, "emitted_s": 1.0},
            {"category": "possible_fall", "near_miss": True, "start_s": 2.0, "emitted_s": 2.4},
        ],
    }

    def fake_analysis(
        video: Path,
        out_dir: Path,
        manifest: Path,
        time_scale: float = 1.0,
        pad_end_s: float = 0.0,
        camera_height_m: float = 2.5,
        crop_px: tuple[int, int, int, int] | None = None,
        zones_file: Path | None = None,
    ) -> Path:
        out_dir.mkdir(parents=True, exist_ok=True)
        events = out_dir / "events.jsonl"
        lines = [json.dumps(e) for e in alerts[video.parent.name]]
        events.write_text("".join(line + "\n" for line in lines))
        return events

    monkeypatch.setattr(evaluate, "run_analysis", fake_analysis)
    out = tmp_path / "report"
    code = main(
        ["eval", "--dataset", "gmdcsa24", "--root", str(tmp_path / "data"), "--out-dir", str(out)]
    )
    assert code == 0
    report = json.loads((out / "report.json").read_text())
    assert report["falls"] == 1 and report["detected"] == 1
    assert report["false_alerts"] == 1  # the ADL alert; the near-miss is not notified
    assert report["latency_median_s"] == pytest.approx(1.5)
    assert report["false"] == ["Subject 1/ADL/01.mp4"]
    assert "recall" in (out / "report.md").read_text()
    assert "1/1" in capsys.readouterr().out


def test_eval_can_score_again_without_analysing_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gmdcsa24(tmp_path / "data")
    calls: list[Path] = []

    def fake_analysis(
        video: Path,
        out_dir: Path,
        manifest: Path,
        time_scale: float = 1.0,
        pad_end_s: float = 0.0,
        camera_height_m: float = 2.5,
        crop_px: tuple[int, int, int, int] | None = None,
        zones_file: Path | None = None,
    ) -> Path:
        calls.append(video)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "events.jsonl").write_text("")
        return out_dir / "events.jsonl"

    monkeypatch.setattr(evaluate, "run_analysis", fake_analysis)
    command = ["eval", "--dataset", "gmdcsa24", "--root", str(tmp_path / "data")]
    command += ["--out-dir", str(tmp_path / "report")]
    assert main(command) == 0 and len(calls) == 2
    assert main([*command, "--reuse"]) == 0 and len(calls) == 2


def test_reuse_analyses_again_when_a_saved_analysis_is_incomplete_or_for_another_video(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gmdcsa24(tmp_path / "data")
    calls: list[str] = []

    def fake_analysis(
        video: Path,
        out_dir: Path,
        manifest: Path,
        time_scale: float = 1.0,
        pad_end_s: float = 0.0,
        camera_height_m: float = 2.5,
        crop_px: tuple[int, int, int, int] | None = None,
        zones_file: Path | None = None,
    ) -> Path:
        calls.append(video.parent.name)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "events.jsonl").write_text("")
        return out_dir / "events.jsonl"

    monkeypatch.setattr(evaluate, "run_analysis", fake_analysis)
    out = tmp_path / "report"
    command = ["eval", "--dataset", "gmdcsa24", "--root", str(tmp_path / "data")]
    command += ["--out-dir", str(out), "--reuse"]
    assert main(command) == 0 and sorted(calls) == ["ADL", "Fall"]
    folders = sorted(p for p in (out / "videos").iterdir())
    (folders[0] / "complete").unlink()  # an analysis interrupted half-way
    meta = folders[1] / "meta.json"
    meta.write_text(meta.read_text().replace("01.mp4", "02.mp4"))  # another video's folder
    assert main(command) == 0 and len(calls) == 4


def test_mcfd_is_analysed_at_its_capture_rate_not_its_declared_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The MCFD files declare 120 fps; the cameras captured about 30 (#69)."""
    root = tmp_path / "data"
    (root / "dataset/chute24").mkdir(parents=True)
    write_synthetic_video(root / "dataset/chute24/cam1.avi", frames=30, fps=120)
    scales: list[float] = []

    def fake_analysis(
        video: Path,
        out_dir: Path,
        manifest: Path,
        time_scale: float,
        pad_end_s: float = 0.0,
        camera_height_m: float = 2.5,
        crop_px: tuple[int, int, int, int] | None = None,
        zones_file: Path | None = None,
    ) -> Path:
        scales.append(time_scale)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "events.jsonl").write_text("")
        return out_dir / "events.jsonl"

    monkeypatch.setattr(evaluate, "run_analysis", fake_analysis)
    out = tmp_path / "report"
    assert main(["eval", "--dataset", "mcfd", "--root", str(root), "--out-dir", str(out)]) == 0
    assert scales == [pytest.approx(4.0)]
    report = json.loads((out / "report.json").read_text())
    assert report["hours"] * 3600 == pytest.approx(30 / 30, abs=0.05)  # 30 frames at 30 fps


def test_eval_pads_each_clip_by_a_few_seconds_and_reports_late_alerts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Clips end a few seconds after the fall: the analysis goes on over the last frame
    repeated for 5 s, enough for pending fall decisions; repeated for the whole deadline,
    the frozen frame was a person lying still long enough for person_down (#97)."""
    gmdcsa24(tmp_path / "data")
    pads: list[float] = []

    def fake_analysis(
        video: Path,
        out_dir: Path,
        manifest: Path,
        time_scale: float = 1.0,
        pad_end_s: float = 0.0,
        camera_height_m: float = 2.5,
        crop_px: tuple[int, int, int, int] | None = None,
        zones_file: Path | None = None,
    ) -> Path:
        pads.append(pad_end_s)
        out_dir.mkdir(parents=True, exist_ok=True)
        events = out_dir / "events.jsonl"
        late = {"category": "person_down", "near_miss": False, "start_s": 1.5, "emitted_s": 40.0}
        events.write_text(json.dumps(late) + "\n" if video.parent.name == "Fall" else "")
        return events

    monkeypatch.setattr(evaluate, "run_analysis", fake_analysis)
    out = tmp_path / "report"
    args = ["eval", "--dataset", "gmdcsa24", "--root", str(tmp_path / "data"), "--out-dir"]
    assert main([*args, str(out), "--deadline-s", "30"]) == 0
    assert pads == [PAD_END_S, PAD_END_S] == [5.0, 5.0]
    report = json.loads((out / "report.json").read_text())
    assert report["detected"] == 0 and report["detected_within_late_deadline"] == 1
    markdown = (out / "report.md").read_text()
    assert "within 30 s" in markdown and "within 60 s" in markdown
    assert "on padding" in markdown


def test_eval_can_analyse_with_the_safe_zones_suggested_from_each_video(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--safe-zones suggested: furniture found in each video's first frame becomes accepted
    safe zones over its top, as an installer would accept them (#106)."""
    gmdcsa24(tmp_path / "data")
    furniture = [("bed", (64.0, 36.0, 320.0, 180.0)), ("sofa", (400.0, 200.0, 600.0, 300.0))]
    monkeypatch.setattr(evaluate, "furniture_detector", lambda args: lambda image: furniture)
    seen: list[Path | None] = []

    def fake_analysis(
        video: Path,
        out_dir: Path,
        manifest: Path,
        time_scale: float = 1.0,
        pad_end_s: float = 0.0,
        camera_height_m: float = 2.5,
        crop_px: tuple[int, int, int, int] | None = None,
        zones_file: Path | None = None,
    ) -> Path:
        seen.append(zones_file)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "events.jsonl").write_text("")
        return out_dir / "events.jsonl"

    monkeypatch.setattr(evaluate, "run_analysis", fake_analysis)
    out = tmp_path / "report"
    args = ["eval", "--dataset", "gmdcsa24", "--root", str(tmp_path / "data"), "--out-dir"]
    assert main([*args, str(out), "--safe-zones", "suggested"]) == 0
    assert len(seen) == 2 and all(path is not None for path in seen)
    zones = load_zones(seen[0])  # type: ignore[arg-type]
    assert [(z.name, z.kind) for z in zones] == [("bed_1", "safe"), ("sofa_1", "safe")]
    assert (out / "report.safe-zones.json").exists()


def test_the_mcfd_mattress_is_never_a_safe_zone() -> None:
    """The mattress on the floor is where the falls of the Multiple Cameras Fall Dataset
    happen; the detector sees it as a bed."""
    assert evaluate.SAFE_ZONE_FURNITURE["mcfd"] == ("sofa",)
    assert evaluate.SAFE_ZONE_FURNITURE.get("gmdcsa24", evaluate.DEFAULT_SAFE_ZONE_FURNITURE) == (
        "bed",
        "sofa",
    )
