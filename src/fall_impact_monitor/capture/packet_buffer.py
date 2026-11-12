"""In-memory buffer of recent encoded video packets for on-demand re-decoding.

The confirmation stage re-decodes a short window at full frame rate. Only
encoded packets are kept (a few MB for ~20 s), never decoded frames.
"""

import threading
from bisect import bisect_left, bisect_right
from collections import deque
from collections.abc import Iterator
from typing import Any

import av
from av.video.codeccontext import VideoCodecContext
from av.video.frame import VideoFrame

from fall_impact_monitor.capture.timing import FrameRecord


class PacketBuffer:
    """Keeps packets whose arrival time is within ``window_s`` of the newest one.
    Safe to append from the reader thread while another thread reads a window."""

    def __init__(self, window_s: float, codec_name: str, extradata: bytes | None) -> None:
        if window_s <= 0:
            raise ValueError("window_s must be positive")
        self._window_ns = int(window_s * 1e9)
        self._codec_name = codec_name
        self._extradata = extradata
        self._items: deque[tuple[FrameRecord, bytes]] = deque()
        self._lock = threading.Lock()

    def __len__(self) -> int:
        return len(self._items)

    def append(self, record: FrameRecord, payload: bytes) -> None:
        with self._lock:
            self._items.append((record, payload))
            cutoff_ns = record.arrival_ns - self._window_ns
            while self._items and self._items[0][0].arrival_ns < cutoff_ns:
                self._items.popleft()

    def newest_arrival_ns(self) -> int | None:
        with self._lock:
            return self._items[-1][0].arrival_ns if self._items else None

    def span_ns(self) -> int:
        with self._lock:
            if not self._items:
                return 0
            return self._items[-1][0].arrival_ns - self._items[0][0].arrival_ns

    def window(self, start_ns: int, end_ns: int) -> list[tuple[FrameRecord, bytes]]:
        """Packets arriving in ``[start_ns, end_ns]``, extended back to the previous keyframe."""
        with self._lock:
            items = list(self._items)
        arrivals = [record.arrival_ns for record, _ in items]
        first = bisect_left(arrivals, start_ns)
        last = bisect_right(arrivals, end_ns)
        if first >= last:
            return []
        while first > 0 and not items[first][0].keyframe:
            first -= 1
        return items[first:last]

    def decode_window(self, start_ns: int, end_ns: int) -> Iterator[tuple[FrameRecord, VideoFrame]]:
        """Decode the window with a fresh decoder; yield frames whose record is inside it.

        Decoding starts at the first keyframe; packets before it cannot be decoded.
        The decoder's PTS carries the item index, only to map frames back to records.
        """
        items = self.window(start_ns, end_ns)
        start = next((i for i, (record, _) in enumerate(items) if record.keyframe), None)
        if start is None:
            return
        decoder = av.CodecContext.create(self._codec_name, "r")
        if not isinstance(decoder, VideoCodecContext):
            raise ValueError(f"{self._codec_name} is not a video codec")
        if self._extradata:
            decoder.extradata = self._extradata
        for index, (_, payload) in enumerate(items[start:], start=start):
            packet = av.Packet(payload)
            packet.pts = packet.dts = index
            yield from self._inside(decoder.decode(packet), items, start_ns, end_ns)
        yield from self._inside(decoder.decode(None), items, start_ns, end_ns)

    @staticmethod
    def _inside(
        frames: list[VideoFrame],
        items: list[tuple[FrameRecord, bytes]],
        start_ns: int,
        end_ns: int,
    ) -> Iterator[tuple[FrameRecord, VideoFrame]]:
        for frame in frames:
            if frame.pts is None:
                continue
            record = items[frame.pts][0]
            if start_ns <= record.arrival_ns <= end_ns:
                yield record, frame


class LivePacketBuffer:
    """The packet buffer of one camera's analysis stream, fed by its reader thread.
    Created on the first packet (codec and extradata come from the stream) and dropped
    on reset (a reconnection may bring different codec parameters)."""

    def __init__(self, window_s: float) -> None:
        self._window_s = window_s
        self._buffer: PacketBuffer | None = None

    def add(self, record: FrameRecord, packet: Any) -> None:
        buffer = self._buffer
        if buffer is None:
            context = packet.stream.codec_context
            buffer = self._buffer = PacketBuffer(self._window_s, context.name, context.extradata)
        buffer.append(record, bytes(packet))

    def reset(self) -> None:
        self._buffer = None

    def newest_arrival_ns(self) -> int | None:
        buffer = self._buffer
        return buffer.newest_arrival_ns() if buffer is not None else None

    def decode_window(self, start_ns: int, end_ns: int) -> Iterator[tuple[FrameRecord, VideoFrame]]:
        buffer = self._buffer
        if buffer is not None:
            yield from buffer.decode_window(start_ns, end_ns)
