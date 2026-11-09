import json
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pytest

from fall_impact_monitor.recording import Span
from fall_impact_monitor.recording.clip_media import max_clip_duration_s, probe
from fall_impact_monitor.recording.clips import (
    ClipUnavailable,
    EventClipExporter,
    event_clip_path,
    export_clip,
)
from tests.conftest import FakePlayback
from tests.recording.media import write_camera_clip

T0 = datetime(2026, 11, 9, 3, 0, 0, tzinfo=UTC)
CLIP_S = 10.5  # what the fake playback server returns: 0.5 s before the start (keyframe)


@pytest.fixture(scope="module")
def camera_clip(tmp_path_factory: pytest.TempPathFactory) -> bytes:
    path = write_camera_clip(tmp_path_factory.mktemp("media") / "camera.mov", CLIP_S)
    return path.read_bytes()


def serve(playback: FakePlayback, clip: bytes, *spans: tuple[datetime, float]) -> None:
    items = [{"start": s.isoformat(), "duration": d, "url": "x"} for s, d in spans]
    playback.routes["/list"] = lambda q: (200, json.dumps(items).encode()) if items else (404, b"")
    playback.routes["/get"] = lambda q: (200, clip)


def test_full_coverage_gives_the_requested_window_snapped_to_the_keyframe(
    playback: FakePlayback, camera_clip: bytes, tmp_path: Path
) -> None:
    serve(playback, camera_clip, (T0 - timedelta(minutes=1), 600.0))
    dest = tmp_path / "clip.mp4"
    result = export_clip(
        playback.url, "cam1", T0, T0 + timedelta(seconds=10), dest, max_bytes=50_000_000
    )
    assert result.requested_start == T0
    assert result.requested_end == T0 + timedelta(seconds=10)
    assert result.actual_end == T0 + timedelta(seconds=10)
    assert (T0 - result.actual_start).total_seconds() == pytest.approx(0.5, abs=0.2)
    assert result.gaps == []
    assert result.has_audio and probe(dest).audio_codec == "aac"
    sidecar = json.loads(dest.with_suffix(".json").read_text(encoding="utf-8"))
    assert sidecar["requested_start"] == "2026-11-09T03:00:00+00:00"
    assert sidecar["gaps"] == []
    path, query = playback.requests[-1]
    assert path == "/get"
    assert query["start"] == "2026-11-09T03:00:00Z"
    assert query["duration"] == "10"
    assert query["format"] == "mp4"


def test_a_recording_that_stops_early_reports_the_gap(
    playback: FakePlayback, camera_clip: bytes, tmp_path: Path
) -> None:
    serve(playback, camera_clip, (T0 - timedelta(minutes=1), 70.0))
    end = T0 + timedelta(seconds=40)
    result = export_clip(playback.url, "cam1", T0, end, tmp_path / "c.mp4", max_bytes=50_000_000)
    assert result.actual_end == T0 + timedelta(seconds=10)
    assert result.gaps == [Span(T0 + timedelta(seconds=10), end)]


def test_a_window_starting_in_a_gap_is_requested_from_the_first_recorded_instant(
    playback: FakePlayback, camera_clip: bytes, tmp_path: Path
) -> None:
    first = T0 + timedelta(seconds=5)
    serve(playback, camera_clip, (first, 600.0))

    def get(query: dict[str, str]) -> tuple[int, bytes]:
        # MediaMTX answers 404 when the requested start falls in a gap.
        requested = datetime.fromisoformat(query["start"])
        return (404, b"no recording segments found") if requested < first else (200, camera_clip)

    playback.routes["/get"] = get
    end = T0 + timedelta(seconds=15)
    result = export_clip(playback.url, "cam1", T0, end, tmp_path / "c.mp4", max_bytes=50_000_000)
    _, query = playback.requests[-1]
    assert query["start"] == "2026-11-09T03:00:05Z"
    assert query["duration"] == "10"
    assert result.gaps == [Span(T0, first)]
    assert result.actual_start <= first


def test_no_recording_in_the_window_writes_nothing(
    playback: FakePlayback, camera_clip: bytes, tmp_path: Path
) -> None:
    serve(playback, camera_clip, (T0 - timedelta(hours=1), 60.0))
    with pytest.raises(ClipUnavailable):
        export_clip(
            playback.url,
            "cam1",
            T0,
            T0 + timedelta(seconds=10),
            tmp_path / "c.mp4",
            max_bytes=50_000_000,
        )
    assert list(tmp_path.iterdir()) == []


def test_a_failed_download_leaves_no_clip_and_no_sidecar(
    playback: FakePlayback, camera_clip: bytes, tmp_path: Path
) -> None:
    serve(playback, camera_clip, (T0 - timedelta(minutes=1), 600.0))
    playback.routes["/get"] = lambda q: (500, b"error")
    with pytest.raises(Exception):  # noqa: B017 - the HTTP error propagates
        export_clip(
            playback.url,
            "cam1",
            T0,
            T0 + timedelta(seconds=10),
            tmp_path / "c.mp4",
            max_bytes=50_000_000,
        )
    assert list(tmp_path.iterdir()) == []


def test_naive_times_are_rejected(playback: FakePlayback, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="timezone"):
        export_clip(
            playback.url,
            "cam1",
            datetime(2026, 11, 9),
            datetime(2026, 11, 9, 0, 1),
            tmp_path / "c.mp4",
            max_bytes=1,
        )


def test_event_clip_path_is_organized_by_camera_and_utc_date(tmp_path: Path) -> None:
    local = datetime(2026, 11, 9, 23, 30, 5, tzinfo=timezone(timedelta(hours=-3)))
    assert event_clip_path(tmp_path, "cam1", local, "abc") == (
        tmp_path / "cam1" / "2026" / "11" / "10" / "023005_abc.mp4"
    )


def test_event_clip_requests_the_pre_and_post_window(
    playback: FakePlayback, camera_clip: bytes, tmp_path: Path
) -> None:
    serve(playback, camera_clip, (T0 - timedelta(minutes=5), 900.0))
    now = [T0 + timedelta(seconds=200)]
    exporter = EventClipExporter(
        playback.url,
        tmp_path,
        pre_s=20,
        post_s=60,
        max_bytes=50_000_000,
        clock=lambda: now[0],
        sleep=lambda s: None,
    )
    result = exporter.export("cam1", "ev1", T0, T0 + timedelta(seconds=5))
    assert result.requested_start == T0 - timedelta(seconds=20)
    assert result.requested_end == T0 + timedelta(seconds=65)
    assert result.path == tmp_path / "cam1" / "2026" / "11" / "09" / "030000_ev1.mp4"
    _, query = playback.requests[-1]
    assert query["start"] == "2026-11-09T02:59:40Z"
    assert query["duration"] == "85"


def test_each_recorded_piece_is_requested_and_joined(
    playback: FakePlayback, camera_clip: bytes, tmp_path: Path
) -> None:
    # MediaMTX's /get stops at the first gap: one request per recorded piece.
    serve(
        playback,
        camera_clip,
        (T0 - timedelta(seconds=30), 40.0),
        (T0 + timedelta(seconds=20), 600.0),
    )
    end = T0 + timedelta(seconds=30)
    result = export_clip(playback.url, "cam1", T0, end, tmp_path / "c.mp4", max_bytes=50_000_000)
    gets = [q for path, q in playback.requests if path == "/get"]
    assert [(q["start"], q["duration"]) for q in gets] == [
        ("2026-11-09T03:00:00Z", "10"),
        ("2026-11-09T03:00:20Z", "10"),
    ]
    assert probe(result.path).duration_s == pytest.approx(2 * CLIP_S, abs=0.5)
    assert result.gaps == [Span(T0 + timedelta(seconds=10), T0 + timedelta(seconds=20))]
    assert result.transcoded


def test_a_window_too_long_for_the_size_cap_is_truncated(
    playback: FakePlayback, camera_clip: bytes, tmp_path: Path
) -> None:
    serve(playback, camera_clip, (T0 - timedelta(minutes=1), 600.0))
    max_bytes = 200_000  # at the minimum bitrate this holds about 4.6 s
    result = export_clip(
        playback.url,
        "cam1",
        T0,
        T0 + timedelta(seconds=10),
        tmp_path / "c.mp4",
        max_bytes=max_bytes,
    )
    _, query = playback.requests[-1]
    assert float(query["duration"]) == pytest.approx(max_clip_duration_s(max_bytes), abs=0.01)
    assert result.truncated
    assert result.requested_end == T0 + timedelta(seconds=10)
    assert result.actual_end <= T0 + timedelta(seconds=4.6)
    assert json.loads(result.path.with_suffix(".json").read_text())["truncated"] is True
