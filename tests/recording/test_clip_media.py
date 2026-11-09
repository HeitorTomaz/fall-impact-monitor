from pathlib import Path

import pytest

from fall_impact_monitor.recording.clip_media import (
    fit_clip,
    max_clip_duration_s,
    probe,
    remux_aac,
)
from tests.recording.media import write_camera_clip


def test_remux_converts_g711_audio_to_aac_and_copies_the_video(tmp_path: Path) -> None:
    src = write_camera_clip(tmp_path / "camera.mov", 4.0)
    assert probe(src).audio_codec == "pcm_mulaw"
    dest = tmp_path / "clip.mp4"
    remux_aac(src, dest)
    info = probe(dest)
    assert info.video_codec == "h264"
    assert info.audio_codec == "aac"
    assert info.duration_s == pytest.approx(4.0, abs=0.2)
    assert info.size_bytes == dest.stat().st_size


def test_a_clip_without_audio_stays_video_only(tmp_path: Path) -> None:
    src = write_camera_clip(tmp_path / "camera.mov", 2.0, audio=False)
    dest = tmp_path / "clip.mp4"
    remux_aac(src, dest)
    assert probe(dest).audio_codec is None


def test_a_small_clip_is_only_remuxed(tmp_path: Path) -> None:
    src = write_camera_clip(tmp_path / "camera.mov", 2.0)
    assert fit_clip([src], tmp_path / "clip.mp4", max_bytes=50_000_000) is False


def test_a_clip_above_the_cap_is_reencoded_below_it(tmp_path: Path) -> None:
    src = write_camera_clip(tmp_path / "noise.mov", 5.0, noise=True)
    max_bytes = 400_000
    assert src.stat().st_size > 4 * max_bytes
    dest = tmp_path / "clip.mp4"
    assert fit_clip([src], dest, max_bytes=max_bytes) is True
    info = probe(dest)
    assert info.size_bytes <= max_bytes
    assert info.audio_codec == "aac"
    assert info.duration_s == pytest.approx(5.0, abs=0.3)


def test_a_failure_leaves_no_file_behind(tmp_path: Path) -> None:
    src = tmp_path / "corrupt.mp4"
    src.write_bytes(b"not a video" * 100)
    dest = tmp_path / "clip.mp4"
    with pytest.raises(Exception):  # noqa: B017 - any decoding error
        fit_clip([src], dest, max_bytes=50_000_000)
    assert list(tmp_path.iterdir()) == [src]


def test_recorded_pieces_are_joined_into_one_clip(tmp_path: Path) -> None:
    first = write_camera_clip(tmp_path / "a.mov", 4.0)
    second = write_camera_clip(tmp_path / "b.mov", 3.0)
    dest = tmp_path / "clip.mp4"
    assert fit_clip([first, second], dest, max_bytes=50_000_000) is True  # re-encoded
    info = probe(dest)
    assert info.video_codec == "h264" and info.audio_codec == "aac"
    assert info.duration_s == pytest.approx(7.0, abs=0.3)


def test_pieces_with_different_sizes_are_joined_at_the_first_size(tmp_path: Path) -> None:
    first = write_camera_clip(tmp_path / "a.mov", 2.0)
    second = write_camera_clip(tmp_path / "b.mov", 2.0, size=(160, 120))
    dest = tmp_path / "clip.mp4"
    fit_clip([first, second], dest, max_bytes=50_000_000)
    assert probe(dest).duration_s == pytest.approx(4.0, abs=0.3)


def test_the_longest_duration_that_fits_uses_the_minimum_bitrate() -> None:
    assert max_clip_duration_s(50_000_000) == pytest.approx(0.9 * 50e6 * 8 / 314_000, rel=1e-6)
