"""PyAV-based readers for files and network streams.

`iter_frame_records` reads one source once. `ReconnectingReader` wraps it for
long-running streams: it reconnects with exponential backoff, reports stale
streams, and masks credentials in every message.
"""

import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from fractions import Fraction
from typing import Any, Literal

import av
from av.video.stream import VideoStream

from fall_impact_monitor.capture.timing import FrameRecord, TimestampClassifier
from fall_impact_monitor.config import mask_url

Opener = Callable[..., Any]
_NETWORK_SCHEMES = (
    "rtsp://",
    "rtsps://",
    "rtmp://",
    "http://",
    "https://",
    "udp://",
    "tcp://",
    "srt://",
)


@dataclass(frozen=True)
class SourceOptions:
    """How a source is opened."""

    video_only: bool = True
    wallclock: bool = False
    open_timeout_s: float = 10.0
    read_timeout_s: float = 5.0


@dataclass(frozen=True)
class FrameEvent:
    record: FrameRecord
    packet: "av.Packet[VideoStream]"


@dataclass(frozen=True)
class StateEvent:
    state: Literal["connected", "stale", "disconnected", "ended"]
    detail: str


ReaderEvent = FrameEvent | StateEvent


def is_network_source(url: str) -> bool:
    return url.lower().startswith(_NETWORK_SCHEMES)


def demux_options(url: str, opts: SourceOptions) -> dict[str, str]:
    """FFmpeg demuxer options for ``url``.

    For RTSP, ``allowed_media_types=video`` sets up a single-stream session so
    FFmpeg derives PTS from RTP timestamps only, without RTCP-based
    cross-stream synchronization.
    """
    options: dict[str, str] = {}
    if url.lower().startswith(("rtsp://", "rtsps://")):
        options["rtsp_transport"] = "tcp"
        if opts.video_only:
            options["allowed_media_types"] = "video"
    if opts.wallclock:
        options["use_wallclock_as_timestamps"] = "1"
    return options


def iter_frame_records(
    url: str,
    camera_id: str,
    opts: SourceOptions,
    *,
    clock_ns: Callable[[], int] = time.monotonic_ns,
    opener: Opener = av.open,
) -> Iterator[tuple[FrameRecord, "av.Packet[VideoStream]"]]:
    """Yield a `FrameRecord` and the encoded packet for every video packet of ``url``."""
    container = opener(
        url,
        options=demux_options(url, opts),
        timeout=(opts.open_timeout_s, opts.read_timeout_s),
    )
    try:
        stream = container.streams.video[0]
        classifier = TimestampClassifier()
        seq = 0
        for packet in container.demux(stream):
            if not packet.size:
                continue  # demuxer flush packet
            arrival_ns = clock_ns()
            time_base = Fraction(packet.time_base or stream.time_base)
            record = FrameRecord(
                camera_id=camera_id,
                seq=seq,
                pts=packet.pts,
                time_base=time_base,
                arrival_ns=arrival_ns,
                wall_utc=datetime.now(UTC),
                keyframe=bool(packet.is_keyframe),
                size_bytes=packet.size,
                quality=classifier.classify(packet.pts, time_base),
            )
            seq += 1
            yield record, packet
    finally:
        container.close()


class ReconnectingReader:
    """Iterate `ReaderEvent`s from a source, reconnecting network streams forever.

    A stream that raises after ``stale_timeout_s`` without frames is reported
    as ``stale``; other failures as ``disconnected``. File sources end with
    ``ended`` and are not reopened.
    """

    def __init__(
        self,
        url: str,
        camera_id: str,
        opts: SourceOptions,
        *,
        backoff_initial_s: float = 1.0,
        backoff_max_s: float = 30.0,
        stale_timeout_s: float = 10.0,
        opener: Opener = av.open,
        sleep: Callable[[float], None] = time.sleep,
        clock_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        self._url = url
        self._camera_id = camera_id
        self._opts = opts
        self._backoff_initial_s = backoff_initial_s
        self._backoff_max_s = backoff_max_s
        self._stale_timeout_ns = int(stale_timeout_s * 1e9)
        self._opener = opener
        self._sleep = sleep
        self._clock_ns = clock_ns
        self._closed = False

    def close(self) -> None:
        self._closed = True

    def __iter__(self) -> Iterator[ReaderEvent]:
        backoff_s = self._backoff_initial_s
        network = is_network_source(self._url)
        while not self._closed:
            last_frame_ns: int | None = None
            connected = False
            try:
                records = iter_frame_records(
                    self._url,
                    self._camera_id,
                    self._opts,
                    clock_ns=self._clock_ns,
                    opener=self._opener,
                )
                for record, packet in records:
                    if not connected:
                        connected = True
                        backoff_s = self._backoff_initial_s
                        yield StateEvent("connected", mask_url(self._url))
                    last_frame_ns = record.arrival_ns
                    yield FrameEvent(record, packet)
                    if self._closed:
                        return
                if not network:
                    yield StateEvent("ended", mask_url(self._url))
                    return
                yield StateEvent("disconnected", "end of stream")
            except Exception as exc:
                detail = mask_url(f"{type(exc).__name__}: {exc}")
                silent_ns = self._clock_ns() - (last_frame_ns if last_frame_ns is not None else 0)
                stale = last_frame_ns is not None and silent_ns >= self._stale_timeout_ns
                yield StateEvent("stale" if stale else "disconnected", detail)
                if not network:
                    return
            if self._closed:
                return
            self._sleep(backoff_s)
            backoff_s = min(backoff_s * 2, self._backoff_max_s)
