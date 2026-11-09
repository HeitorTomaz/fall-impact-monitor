"""Clip media processing with PyAV: one MP4 with AAC audio, under a size cap.

Cameras often send G711 audio, which most players and Telegram do not play in MP4, so
audio is always re-encoded to AAC. A clip made of one recorded piece keeps its video
unchanged. A clip made of several pieces (the recording had gaps), or one above the size
cap, is re-encoded with H.264: the pieces are joined back to back (the gaps are cut out
and listed by the caller), each frame keeping its time within its piece, and the audio
is padded with silence where a piece has less audio than video, so it stays in sync.

Every function writes to a temporary file first: on error nothing is left at the
destination.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

import av
import numpy as np

AUDIO_BPS = 64_000
MIN_VIDEO_BPS = 250_000  # below this a re-encoded clip is not worth watching
_SIZE_MARGIN = 0.9  # target this share of the cap (container overhead, rate control)
_RETRY_FACTOR = 0.7  # bitrate factor for the second attempt when still too large
_CRF = "23"  # quality of a re-encode that is not size-constrained
_TIME_BASE = Fraction(1, 90_000)


@dataclass(frozen=True)
class MediaInfo:
    duration_s: float
    video_codec: str
    audio_codec: str | None
    size_bytes: int


def probe(path: Path) -> MediaInfo:
    with av.open(str(path)) as container:
        video = container.streams.video[0]
        audio = container.streams.audio[0] if container.streams.audio else None
        if container.duration is not None:
            duration_s = container.duration / 1_000_000
        else:
            duration_s = float((video.duration or 0) * video.time_base) if video.time_base else 0.0
        return MediaInfo(
            duration_s=duration_s,
            video_codec=video.codec_context.name,
            audio_codec=audio.codec_context.name if audio is not None else None,
            size_bytes=path.stat().st_size,
        )


def max_clip_duration_s(max_bytes: int) -> float:
    """The longest clip that fits ``max_bytes`` at the minimum video bitrate."""
    return _SIZE_MARGIN * max_bytes * 8 / (MIN_VIDEO_BPS + AUDIO_BPS)


@contextmanager
def _atomic(dest: Path) -> Iterator[Path]:
    partial = dest.with_name(dest.stem + ".part.mp4")
    try:
        yield partial
        partial.replace(dest)
    finally:
        partial.unlink(missing_ok=True)


def _open_output(path: Path) -> Any:
    return av.open(str(path), "w", format="mp4", options={"movflags": "+faststart"})


def _add_aac(out: Any, rate: int, channels: int) -> tuple[Any, str]:
    layout = "mono" if channels == 1 else "stereo"
    stream = out.add_stream("aac", rate=rate, layout=layout)
    stream.bit_rate = AUDIO_BPS
    return stream, layout


def _copy_video(src: Path, partial: Path) -> None:
    """One piece: video copied, audio re-encoded to AAC."""
    with av.open(str(src)) as inp, _open_output(partial) as out:
        vin = inp.streams.video[0]
        ain = inp.streams.audio[0] if inp.streams.audio else None
        vout = out.add_stream_from_template(vin)
        aout: Any = None
        resampler = None
        if ain is not None:
            aout, layout = _add_aac(out, ain.rate, ain.channels)
            resampler = av.AudioResampler(format="fltp", layout=layout, rate=ain.rate)
        for packet in inp.demux(*([vin] if ain is None else [vin, ain])):
            if packet.dts is None:  # flush packet
                continue
            if packet.stream is vin:
                packet.stream = vout
                out.mux(packet)
            elif resampler is not None:
                for frame in packet.decode():
                    if isinstance(frame, av.AudioFrame):
                        for resampled in resampler.resample(frame):
                            out.mux(aout.encode(resampled))
        if resampler is not None:
            for resampled in resampler.resample(None):
                out.mux(aout.encode(resampled))
            out.mux(aout.encode())


def _encode(srcs: list[Path], partial: Path, video_bps: int | None) -> None:
    """Pieces joined back to back and re-encoded (H.264 at ``video_bps``, or constant
    quality when None), at the first piece's size."""
    with av.open(str(srcs[0])) as first:
        vfirst = first.streams.video[0]
        width, height = vfirst.width, vfirst.height
        rate = vfirst.average_rate or 15
        afirst = first.streams.audio[0] if first.streams.audio else None
        audio_rate = afirst.rate if afirst is not None else 0
        channels = afirst.channels if afirst is not None else 0
    with _open_output(partial) as out:
        vout = out.add_stream("libx264", rate=rate)
        vout.width, vout.height, vout.pix_fmt = width, height, "yuv420p"
        vout.codec_context.time_base = _TIME_BASE
        if video_bps is None:
            vout.options = {"preset": "veryfast", "crf": _CRF}
        else:
            vout.bit_rate = video_bps
            vout.options = {
                "preset": "veryfast",
                "maxrate": str(video_bps),
                "bufsize": str(2 * video_bps),
            }
        aout: Any = None
        layout = ""
        if audio_rate:
            aout, layout = _add_aac(out, audio_rate, channels)
        offset_s = 0.0  # start of the current piece in the joined clip
        written = 0  # audio samples written so far
        last_pts = -1
        for src in srcs:
            piece_s = probe(src).duration_s
            if aout is not None:
                written = _silence(out, aout, layout, audio_rate, written, offset_s)
            with av.open(str(src)) as inp:
                vin = inp.streams.video[0]
                ain = inp.streams.audio[0] if inp.streams.audio else None
                resampler = (
                    av.AudioResampler(format="fltp", layout=layout, rate=audio_rate)
                    if aout is not None and ain is not None
                    else None
                )
                origin: Fraction | None = None
                for packet in inp.demux(*([vin] if ain is None else [vin, ain])):
                    if packet.dts is None:
                        continue
                    for frame in packet.decode():
                        if isinstance(frame, av.VideoFrame):
                            if frame.pts is None or frame.time_base is None:
                                continue
                            t = frame.pts * frame.time_base
                            origin = t if origin is None else origin
                            pts = round((Fraction(offset_s) + t - origin) / _TIME_BASE)
                            if pts <= last_pts:  # never out of order across pieces
                                continue
                            last_pts = pts
                            image = frame.reformat(width=width, height=height, format="yuv420p")
                            image.pts, image.time_base = pts, _TIME_BASE
                            out.mux(vout.encode(image))
                        elif isinstance(frame, av.AudioFrame) and resampler is not None:
                            for resampled in resampler.resample(frame):
                                written = _mux_audio(out, aout, resampled, written)
                if resampler is not None:
                    for resampled in resampler.resample(None):
                        written = _mux_audio(out, aout, resampled, written)
            offset_s += piece_s
        out.mux(vout.encode())
        if aout is not None:
            _silence(out, aout, layout, audio_rate, written, offset_s)
            out.mux(aout.encode())


def _mux_audio(out: Any, aout: Any, frame: av.AudioFrame, written: int) -> int:
    frame.pts, frame.time_base = written, Fraction(1, frame.sample_rate)
    out.mux(aout.encode(frame))
    return written + frame.samples


def _silence(out: Any, aout: Any, layout: str, rate: int, written: int, until_s: float) -> int:
    """Pad the audio with silence up to ``until_s`` (keeps it in sync with the video)."""
    missing = round(until_s * rate) - written
    if missing <= 0:
        return written
    channels = 1 if layout == "mono" else 2
    frame = av.AudioFrame.from_ndarray(
        np.zeros((channels, missing), np.float32), format="fltp", layout=layout
    )
    frame.sample_rate = rate
    return _mux_audio(out, aout, frame, written)


def remux_aac(src: Path, dest: Path) -> None:
    """Video copied, audio (any codec) re-encoded to AAC; no audio stays video only."""
    with _atomic(dest) as partial:
        _copy_video(src, partial)


def transcode_to_size(srcs: list[Path], dest: Path, max_bytes: int) -> None:
    """Re-encode at the bitrate that fits ``max_bytes`` (at least ``MIN_VIDEO_BPS``);
    retry once lower when the result is still too large."""
    duration_s = max(sum(probe(s).duration_s for s in srcs), 1.0)
    video_bps = max(int(_SIZE_MARGIN * max_bytes * 8 / duration_s) - AUDIO_BPS, MIN_VIDEO_BPS)
    with _atomic(dest) as partial:
        _encode(srcs, partial, video_bps)
    if dest.stat().st_size > max_bytes:
        with _atomic(dest) as partial:
            _encode(srcs, partial, int(video_bps * _RETRY_FACTOR))


def fit_clip(srcs: list[Path], dest: Path, max_bytes: int) -> bool:
    """One clip from the recorded pieces ``srcs``, at most ``max_bytes`` when the caller
    kept the duration within ``max_clip_duration_s``. Returns whether the video was
    re-encoded (several pieces, or one piece above the cap)."""
    if len(srcs) == 1:
        remux_aac(srcs[0], dest)
        if dest.stat().st_size <= max_bytes:
            return False
    else:
        with _atomic(dest) as partial:
            _encode(srcs, partial, None)
        if dest.stat().st_size <= max_bytes:
            return True
    transcode_to_size(srcs, dest, max_bytes)
    return True
