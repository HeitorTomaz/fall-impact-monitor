"""`fim eval --replay`: the detectors run again on saved analyses, without perception (#67)."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from fall_impact_monitor.api.perception import DetectedPerson, FrameResult
from fall_impact_monitor.cli import analyze
from fall_impact_monitor.cli.analyze import frame_entry, replay, result_from_entry
from fall_impact_monitor.cli.main import main
from fall_impact_monitor.config import Settings
from fall_impact_monitor.perception.pose import Pose
from fall_impact_monitor.perception.tracking import Track
from tests.analysis.test_fall_analyzer import HIP_UP, falling, make_track, standing
from tests.api.test_facade import record
from tests.cli.test_eval_cli import gmdcsa24


def stumble(seq: int) -> list[Track]:
    """Seen live every third frame: a short dip, a near-miss."""
    if 25 <= seq < 35:
        return [make_track(0, HIP_UP + 30.0, 60.0, 0.6)]
    return [standing()]


def fall(seq: int) -> list[Track]:
    """Seen when every frame is analysed again: a fall."""
    if seq < 25:
        return [standing()]
    return [falling(0, min(1.0, (seq - 24) / 10))]


class Scripted:
    def __init__(self, script: Any, every: int) -> None:
        self.script, self.every = script, every

    def process(self, image: Any, rec: Any) -> FrameResult:
        if rec.seq % self.every:
            return FrameResult(rec, [], analysed=False)
        tracks = self.script(rec.seq)
        people = [DetectedPerson(t.box_xyxy, t.score, None, t.track_id) for t in tracks]
        return FrameResult(rec, tracks, analysed=True, detections=people)

    def sharing_models(self, tracker: Any) -> "Scripted":
        return Scripted(fall, every=1)


def without_ids(path: Path) -> list[dict[str, Any]]:
    events = [json.loads(line) for line in path.read_text().splitlines()]
    return [{k: v for k, v in e.items() if k != "event_id"} for e in events]


@pytest.fixture
def scripted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(analyze, "build_pipeline", lambda settings, **kw: Scripted(stumble, 3))


def test_saved_frames_keep_full_precision() -> None:
    keypoints = np.array([[10.123456, 20.987654]] * 17, np.float32)
    scores = np.linspace(0.1, 0.9, 17).astype(np.float32)
    track = Track(
        3, (1.23456789, 2.5, 30.000001, 40.75), 0.876543, keypoints, scores, "stable", 7, 0
    )
    person = DetectedPerson(track.box_xyxy, 0.876543, Pose(keypoints, scores), 3)
    result = FrameResult(record(1.234567891, 12), [track], analysed=True, detections=[person])
    back, size = result_from_entry(json.loads(json.dumps(frame_entry(result, (640, 360)))))
    assert size == (640, 360)
    assert back.record.arrival_ns == result.record.arrival_ns
    assert back.record.seq == 12 and back.record.quality is result.record.quality
    assert back.tracks[0].box_xyxy == track.box_xyxy
    assert back.tracks[0].score == track.score
    assert np.array_equal(back.tracks[0].keypoints, keypoints)  # type: ignore[arg-type]
    assert np.array_equal(back.tracks[0].keypoint_scores, scores)  # type: ignore[arg-type]
    assert back.tracks[0].identity == "stable"
    assert back.detections[0].box_xyxy == person.box_xyxy and back.detections[0].track_id == 3


def test_replay_reproduces_the_analysis_including_its_re_analysis(
    synthetic_video: Path, tmp_path: Path, scripted: None
) -> None:
    assert main(["analyze", "--source", str(synthetic_video), "--out-dir", str(tmp_path)]) == 0
    original = without_ids(tmp_path / "events.jsonl")
    assert any(e["evidence"].get("reanalysed_at_full_rate") for e in original)
    result = replay(tmp_path, Settings())
    assert result.missing_windows == 0
    assert [{k: v for k, v in e.items() if k != "event_id"} for e in result.events] == original


def test_replay_reports_a_candidate_whose_window_was_not_saved(
    synthetic_video: Path, tmp_path: Path, scripted: None
) -> None:
    assert main(["analyze", "--source", str(synthetic_video), "--out-dir", str(tmp_path)]) == 0
    (tmp_path / "confirmations.jsonl").write_text("")
    result = replay(tmp_path, Settings())
    assert result.missing_windows == 1
    assert not any(e["evidence"].get("reanalysed_at_full_rate") for e in result.events)


def run_eval(data: Path, out: Path, *extra: str) -> int:
    return main(
        ["eval", "--dataset", "gmdcsa24", "--root", str(data), "--out-dir", str(out), *extra]
    )


def test_eval_replay_scores_saved_analyses_without_perception(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, scripted: None
) -> None:
    gmdcsa24(tmp_path / "data")
    out = tmp_path / "report"
    assert run_eval(tmp_path / "data", out) == 0

    def no_perception(settings: Any, **kw: Any) -> Any:
        raise AssertionError("replay must not run perception")

    monkeypatch.setattr(analyze, "build_pipeline", no_perception)
    assert run_eval(tmp_path / "data", out, "--replay") == 0
    full = json.loads((out / "report.json").read_text())
    replayed = json.loads((out / "report.replay.json").read_text())
    assert replayed == full
    assert (out / "report.replay.md").exists()


def test_eval_replay_refuses_another_perception_configuration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    scripted: None,
) -> None:
    gmdcsa24(tmp_path / "data")
    out = tmp_path / "report"
    assert run_eval(tmp_path / "data", out) == 0
    monkeypatch.setenv("FIM_ANALYSIS_PERSON_VISIBLE_FPS", "5")
    assert run_eval(tmp_path / "data", out, "--replay") == 2
    assert "FIM_ANALYSIS_PERSON_VISIBLE_FPS" in capsys.readouterr().out
    assert not (out / "report.replay.json").exists()


def test_eval_replay_refuses_analyses_saved_without_their_configuration(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], scripted: None
) -> None:
    gmdcsa24(tmp_path / "data")
    out = tmp_path / "report"
    assert run_eval(tmp_path / "data", out) == 0
    for meta in out.glob("videos/*/meta.json"):
        saved = json.loads(meta.read_text())
        del saved["perception"]
        meta.write_text(json.dumps(saved))
    assert run_eval(tmp_path / "data", out, "--replay") == 2
    assert "fim eval" in capsys.readouterr().out


def test_eval_reuse_analyses_again_when_the_perception_configuration_changed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, scripted: None
) -> None:
    gmdcsa24(tmp_path / "data")
    out = tmp_path / "report"
    assert run_eval(tmp_path / "data", out) == 0
    monkeypatch.setenv("FIM_PERSON_DETECTOR_MAX_FPS", "2")
    assert run_eval(tmp_path / "data", out, "--reuse") == 0
    meta = json.loads(next(out.glob("videos/*/meta.json")).read_text())
    assert meta["perception"]["FIM_PERSON_DETECTOR_MAX_FPS"] == 2.0  # analysed again
