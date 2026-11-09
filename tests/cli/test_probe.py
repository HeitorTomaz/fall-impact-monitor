import csv
import json
from pathlib import Path

import pytest

from fall_impact_monitor.capture.reader import SourceOptions, StateEvent, iter_frame_records
from fall_impact_monitor.cli.main import main
from fall_impact_monitor.cli.probe import write_probe_outputs


def test_probe_writes_csv_json_and_markdown(synthetic_video: Path, tmp_path: Path) -> None:
    assert main(["probe", "--source", str(synthetic_video), "--out-dir", str(tmp_path)]) == 0
    with (tmp_path / "frames.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 75
    assert rows[0]["quality"] == "ok"
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["timing"]["fps_pts"] == 25.0
    assert summary["timing"]["frames"] == 75
    assert "FPS (source PTS)" in (tmp_path / "summary.md").read_text()


def test_outputs_never_contain_credentials(synthetic_video: Path, tmp_path: Path) -> None:
    records = [r for r, _ in iter_frame_records(str(synthetic_video), "cam1", SourceOptions())]
    states = [StateEvent("disconnected", "OSError: rtsp://***@cam.local/s")]
    write_probe_outputs(
        tmp_path, "rtsp://user:secret@cam.local/s", "cam1", SourceOptions(), records, states
    )
    for name in ("frames.csv", "summary.json", "summary.md"):
        assert "secret" not in (tmp_path / name).read_text()
    assert "rtsp://***@cam.local/s" in (tmp_path / "summary.json").read_text()


def test_probe_without_frames_reports_an_error(tmp_path: Path) -> None:
    missing = tmp_path / "missing.mp4"
    assert main(["probe", "--source", str(missing), "--out-dir", str(tmp_path / "out")]) == 1
    summary = json.loads((tmp_path / "out" / "summary.json").read_text())
    assert summary["timing"] is None
    assert summary["states"][0]["state"] == "disconnected"


def test_interrupted_probe_still_writes_what_it_captured(
    synthetic_video: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fall_impact_monitor.capture.reader import FrameEvent
    from fall_impact_monitor.cli import probe

    items = list(iter_frame_records(str(synthetic_video), "cam1", SourceOptions()))[:10]

    class InterruptedReader:
        def __init__(self, *args: object, **kwargs: object) -> None:
            self.kwargs = kwargs

        def __iter__(self):  # type: ignore[no-untyped-def]
            for record, packet in items:
                yield FrameEvent(record, packet)
            raise KeyboardInterrupt

        def close(self) -> None:
            pass

    monkeypatch.setattr(probe, "ReconnectingReader", InterruptedReader)
    with pytest.raises(KeyboardInterrupt):
        main(["probe", "--source", "rtsp://cam.local/s", "--out-dir", str(tmp_path)])
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["timing"]["frames"] == 10


def test_probe_uses_the_configured_stale_timeout(
    synthetic_video: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fall_impact_monitor.cli import probe

    seen: dict[str, object] = {}
    real = probe.ReconnectingReader

    def spy(*args: object, **kwargs: object) -> object:
        seen.update(kwargs)
        return real(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setenv("FIM_STALE_TIMEOUT_S", "3.5")
    monkeypatch.setattr(probe, "ReconnectingReader", spy)
    main(["probe", "--source", str(synthetic_video), "--out-dir", str(tmp_path)])
    assert seen["stale_timeout_s"] == 3.5
