"""Clip media processing with PyAV: MP4 with AAC audio, under a size cap.

Cameras often send G711 audio, which most players and Telegram do not play in MP4, so
audio is always re-encoded to AAC. Video is copied unchanged; only a clip above the
size cap is re-encoded (H.264 at a computed bitrate), keeping every frame's original
timestamp. Every function writes to a temporary file first: on error nothing is left
at the destination.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import av

_AUDIO_BPS = 64_000
_SIZE_MARGIN = 0.9  # target this share of the cap (container overhead, rate control)
_RETRY_FACTOR = 0.7  # bitrate factor for the second attempt when still too large


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


@contextmanager
def _atomic(dest: Path) -> Iterator[Path]:
    partial = dest.with_name(dest.stem + ".part.mp4")
    try:
        yield partial
        partial.replace(dest)
    finally:
        partial.unlink(missing_ok=True)


def _convert(src: Path, dest: Path, video_bps: int | None) -> None:
    """Copy (``video_bps`` None) or re-encode the video; always AAC audio."""
    with (
        _atomic(dest) as partial,
        av.open(str(src)) as inp,
        av.open(str(partial), "w", format="mp4", options={"movflags": "+faststart"}) as out,
    ):
        vin = inp.streams.video[0]
        ain = inp.streams.audio[0] if inp.streams.audio else None
        vout: Any
        if video_bps is None:
            vout = out.add_stream_from_template(vin)
        else:
            vout = out.add_stream("libx264", rate=vin.average_rate)
            vout.width, vout.height, vout.pix_fmt = vin.width, vin.height, "yuv420p"
            vout.codec_context.time_base = vin.time_base
            vout.bit_rate = video_bps
            vout.options = {
                "preset": "veryfast",
                "maxrate": str(video_bps),
                "bufsize": str(2 * video_bps),
            }
        aout: Any = None
        resampler = None
        if ain is not None:
            layout = "mono" if ain.channels == 1 else "stereo"
            aout = out.add_stream("aac", rate=ain.rate, layout=layout)
            aout.bit_rate = _AUDIO_BPS
            resampler = av.AudioResampler(format="fltp", layout=layout, rate=ain.rate)
        streams = [vin] if ain is None else [vin, ain]
        for packet in inp.demux(*streams):
            if packet.dts is None:  # flush packet
                continue
            if packet.stream is vin:
                if video_bps is None:
                    packet.stream = vout
                    out.mux(packet)
                else:
                    for frame in packet.decode():
                        out.mux(vout.encode(frame))
            elif resampler is not None:
                for frame in packet.decode():
                    if not isinstance(frame, av.AudioFrame):
                        continue
                    for resampled in resampler.resample(frame):
                        out.mux(aout.encode(resampled))
        if video_bps is not None:
            out.mux(vout.encode())
        if resampler is not None:
            for resampled in resampler.resample(None):
                out.mux(aout.encode(resampled))
            out.mux(aout.encode())


def remux_aac(src: Path, dest: Path) -> None:
    """Video copied, audio (any codec) re-encoded to AAC; no audio stays video only."""
    _convert(src, dest, None)


def transcode_to_size(src: Path, dest: Path, max_bytes: int) -> None:
    """Re-encode the video at the bitrate that fits ``max_bytes``; retry once lower."""
    info = probe(src)
    duration_s = max(info.duration_s, 1.0)
    audio_bps = _AUDIO_BPS if info.audio_codec is not None else 0
    video_bps = int(_SIZE_MARGIN * max_bytes * 8 / duration_s) - audio_bps
    _convert(src, dest, max(video_bps, 50_000))
    if dest.stat().st_size > max_bytes:
        _convert(src, dest, max(int(video_bps * _RETRY_FACTOR), 50_000))


def fit_clip(src: Path, dest: Path, max_bytes: int) -> bool:
    """Remux; re-encode when the remuxed clip exceeds ``max_bytes``. Returns whether
    it was re-encoded. A clip still above the cap (very long window) is kept: a large
    clip is better than none."""
    remux_aac(src, dest)
    if dest.stat().st_size <= max_bytes:
        return False
    transcode_to_size(src, dest, max_bytes)
    return True
