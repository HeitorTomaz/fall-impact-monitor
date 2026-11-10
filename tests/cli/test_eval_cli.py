import json
from pathlib import Path

import pytest

from fall_impact_monitor.cli import evaluate
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

    def fake_analysis(video: Path, out_dir: Path, manifest: Path) -> Path:
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

    def fake_analysis(video: Path, out_dir: Path, manifest: Path) -> Path:
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

    def fake_analysis(video: Path, out_dir: Path, manifest: Path) -> Path:
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
