import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

from fall_impact_monitor.cli.main import main
from fall_impact_monitor.recording.retention import prune_clips, prune_recordings

UNIT = 100_000  # bytes per synthetic segment
T0 = datetime(2026, 11, 1, 0, 0, tzinfo=UTC)


def segment_name(t: datetime) -> str:
    return t.strftime("%Y-%m-%d_%H-%M-%S-%f_+0000") + ".mp4"


def fill(root: Path, paths: int = 3, segments: int = 20) -> None:
    for p in range(paths):
        folder = root / f"cam{p}"
        folder.mkdir(parents=True)
        for i in range(segments):
            # Paths interleave in time: cam0 minute 0, cam1 minute 0 + 10 s, ...
            t = T0 + timedelta(minutes=i, seconds=10 * p)
            (folder / segment_name(t)).write_bytes(b"\0" * UNIT)


def segments(root: Path) -> list[Path]:
    return sorted(root.glob("*/*.mp4"))


def test_fill_test_never_exceeds_the_cap_and_deletes_the_oldest_first(tmp_path: Path) -> None:
    fill(tmp_path)
    report = prune_recordings(
        tmp_path, max_bytes=10 * UNIT, min_free_bytes=0, free_bytes=lambda: 10**12
    )
    left = segments(tmp_path)
    assert sum(p.stat().st_size for p in left) <= 10 * UNIT
    assert report.deleted_files == 50 and report.freed_bytes == 50 * UNIT
    assert report.recordings_bytes == 10 * UNIT and not report.pressure
    newest = T0 + timedelta(minutes=19)
    for p in range(3):  # the newest segment of every path is kept
        assert (tmp_path / f"cam{p}" / segment_name(newest + timedelta(seconds=10 * p))).exists()
    oldest_left = min(p.name for p in left)
    assert oldest_left >= segment_name(T0 + timedelta(minutes=16))


def test_foreign_and_partial_files_are_never_touched(tmp_path: Path) -> None:
    fill(tmp_path, paths=1, segments=5)
    foreign = [
        tmp_path / "cam0" / "notes.txt",
        tmp_path / "cam0" / "2026-11-01_00-00-00-000000_+0000.mp4.part",
        tmp_path / "cam0" / "not-a-segment.mp4",
        tmp_path / "loose.mp4",
    ]
    for path in foreign:
        path.write_bytes(b"\0" * UNIT)
    prune_recordings(tmp_path, max_bytes=0, min_free_bytes=0, free_bytes=lambda: 10**12)
    assert all(path.exists() for path in foreign)
    assert len(segments(tmp_path)) == 2  # newest segment + not-a-segment.mp4


def test_an_unreachable_cap_reports_pressure(tmp_path: Path) -> None:
    fill(tmp_path, paths=2, segments=3)
    report = prune_recordings(tmp_path, max_bytes=UNIT, min_free_bytes=0, free_bytes=lambda: 10**12)
    assert report.pressure
    assert len(segments(tmp_path)) == 2


def test_low_free_space_deletes_until_the_minimum_is_free(tmp_path: Path) -> None:
    fill(tmp_path, paths=1, segments=10)

    def free() -> int:  # a disk with 12 units, shared with the recordings
        return 12 * UNIT - len(segments(tmp_path)) * UNIT

    report = prune_recordings(tmp_path, max_bytes=10**12, min_free_bytes=5 * UNIT, free_bytes=free)
    assert len(segments(tmp_path)) == 7
    assert report.free_bytes == 5 * UNIT and not report.pressure


def test_dry_run_deletes_nothing(tmp_path: Path) -> None:
    fill(tmp_path, paths=1, segments=5)
    report = prune_recordings(
        tmp_path, max_bytes=2 * UNIT, min_free_bytes=0, free_bytes=lambda: 10**12, dry_run=True
    )
    assert report.deleted_files == 3
    assert len(segments(tmp_path)) == 5


def test_clips_older_than_the_retention_are_deleted_with_their_sidecar(tmp_path: Path) -> None:
    now = datetime(2026, 12, 1, tzinfo=UTC)
    folder = tmp_path / "cam1" / "2026" / "11" / "01"
    folder.mkdir(parents=True)
    for name, age_days in (("old", 31), ("new", 29)):
        for suffix in (".mp4", ".json"):
            path = folder / f"000000_{name}{suffix}"
            path.write_bytes(b"x")
            stamp = (now - timedelta(days=age_days)).timestamp()
            os.utime(path, (stamp, stamp))
    assert prune_clips(tmp_path, max_age_s=30 * 86400, now=now) == 1
    assert sorted(p.name for p in folder.iterdir()) == ["000000_new.json", "000000_new.mp4"]


def test_retention_command_prints_a_report(tmp_path: Path, monkeypatch, capsys) -> None:  # type: ignore[no-untyped-def]
    fill(tmp_path / "rec", paths=1, segments=3)
    (tmp_path / "events").mkdir()
    monkeypatch.setenv("FIM_RECORDINGS_DIR", str(tmp_path / "rec"))
    monkeypatch.setenv("FIM_EVENTS_DIR", str(tmp_path / "events"))
    monkeypatch.setenv("FIM_DISK_MIN_FREE_GB", "0")
    assert main(["recordings", "retention", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "recordings:" in out and "dry run" in out
    assert len(segments(tmp_path / "rec")) == 3
