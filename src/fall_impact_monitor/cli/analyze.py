"""`fim analyze`: run perception and the detectors on a file or stream; write tracks, events
and an optional overlay.

Outputs go to `artifacts/` for local inspection only; never publish them.
"""

import argparse
import dataclasses
import json
import time
from datetime import UTC, datetime
from fractions import Fraction
from pathlib import Path
from typing import Any

import av
import cv2
import numpy as np
from numpy.typing import NDArray

from fall_impact_monitor.analysis.camera import CameraAnalyzer
from fall_impact_monitor.analysis.fall import FallAnalyzer
from fall_impact_monitor.analysis.zones import load_zones
from fall_impact_monitor.api import Track, build_pipeline
from fall_impact_monitor.api.confirmation import FallConfirmer, PendingConfirmations
from fall_impact_monitor.capture.packet_buffer import LivePacketBuffer
from fall_impact_monitor.capture.reader import SourceOptions, is_network_source, iter_frame_records
from fall_impact_monitor.capture.rotation import ROTATIONS_DEG, rotate
from fall_impact_monitor.capture.timing import FrameRecord, TimestampClassifier
from fall_impact_monitor.config import Settings, mask_url
from fall_impact_monitor.events import Event
from fall_impact_monitor.perception.tracking import PersonTracker

# COCO-17 skeleton edges for the overlay.
_EDGES = (
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 6),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (0, 5),
    (0, 6),
)


_CONFIRMATION_TRACKER_FPS = 25.0  # confirmation analyses every frame
_NO_RESIZE_PX = 1_000_000


def add_parser(subparsers: "argparse._SubParsersAction[argparse.ArgumentParser]") -> None:
    parser = subparsers.add_parser("analyze", help="run perception on a file or stream")
    parser.add_argument("--source", required=True)
    parser.add_argument("--camera-id", default="analyze")
    parser.add_argument(
        "--rotate-deg",
        type=int,
        choices=ROTATIONS_DEG,
        default=0,
        help="180 for a camera mounted upside down (as FIM_CAMERA_<ID>_ROTATE_DEG)",
    )
    parser.add_argument("--max-frames", type=int, default=0, help="0 = whole source")
    parser.add_argument("--overlay", action="store_true", help="also write overlay.mp4")
    parser.add_argument("--manifest", type=Path, default=Path("models.toml"))
    parser.add_argument("--zones", type=Path, default=None, help="zone file (see fim zones)")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="output folder (default: artifacts/analyze/<UTC timestamp>)",
    )
    parser.set_defaults(handler=run)


def _track_json(track: Track) -> dict[str, Any]:
    return {
        "id": track.track_id,
        "box": [round(v, 1) for v in track.box_xyxy],
        "score": round(track.score, 3),
        "identity": track.identity,
        "keypoints": None if track.keypoints is None else np.round(track.keypoints, 1).tolist(),
        "keypoint_scores": None
        if track.keypoint_scores is None
        else np.round(track.keypoint_scores, 3).tolist(),
    }


def _event_json(event: Event) -> dict[str, Any]:
    return {
        "event_id": event.event_id,
        "category": event.category.value,
        "confidence": round(event.confidence, 3),
        "near_miss": event.near_miss,
        "start_s": round(event.start_s, 3),
        "end_s": round(event.end_s, 3),
        "track_id": event.track_id,
        "identity": event.identity,
        "evidence": event.evidence,
        "analysed_seqs": event.analysed_seqs,
    }


def draw(image: NDArray[np.uint8], tracks: list[Track]) -> NDArray[np.uint8]:
    canvas = image.copy()
    for track in tracks:
        color = (0, 200, 0) if track.identity == "stable" else (0, 165, 255)
        x1, y1, x2, y2 = (int(v) for v in track.box_xyxy)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            canvas,
            f"{track.track_id} {track.identity}",
            (x1, max(12, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
        )
        if track.keypoints is not None and track.keypoint_scores is not None:
            points = track.keypoints.astype(int)
            for a, b in _EDGES:
                if track.keypoint_scores[a] > 0.3 and track.keypoint_scores[b] > 0.3:
                    cv2.line(canvas, tuple(points[a]), tuple(points[b]), color, 2)
    return canvas


class _Analyzer:
    """Feeds decoded frames, each described by its own presentation timestamp, to the
    pipeline and writes tracks.jsonl and the optional overlay."""

    def __init__(
        self, args: argparse.Namespace, out_dir: Path, tracks_file: Any, events_file: Any
    ) -> None:
        settings = Settings()
        self.pipeline = build_pipeline(
            settings, started_s=time.monotonic(), manifest_path=args.manifest
        )
        self.detectors = CameraAnalyzer(
            args.camera_id,
            load_zones(args.zones) if args.zones else [],
            threshold=settings.alert_min_confidence,
            fall_enabled=settings.fall_detector_enabled,
            floor_enabled=settings.person_down_detector_enabled,
            person_down_s=settings.person_down_detector_after_fall_s,
            no_fall_s=settings.person_down_detector_without_fall_s,
        )
        self.events_file = events_file
        self.events = 0
        self.live = is_network_source(args.source)
        self.rotate_deg: int = args.rotate_deg
        self.overlay_path = out_dir / "overlay.mp4" if args.overlay else None
        self.tracks_file = tracks_file
        self.classifier = TimestampClassifier()
        self.writer: Any = None
        self.stream: Any = None
        self.last_tracks: list[Track] = []
        self.frames = 0
        self.analysed = 0
        self.last_s = 0.0  # time of the last frame read
        self._setup_confirmation(settings, args)

    def _setup_confirmation(self, settings: Settings, args: argparse.Namespace) -> None:
        """Fall near-misses are analysed again at full rate, as in the service."""
        share = getattr(self.pipeline, "sharing_models", None)
        self.pending: PendingConfirmations | None = None
        if not settings.fall_confirmation_enabled or not callable(share):
            return
        zones = load_zones(args.zones) if args.zones else []
        self.before_s = settings.fall_confirmation_before_s
        self.after_s = settings.fall_confirmation_after_s
        self.pending = PendingConfirmations(self.after_s)
        self.packets = LivePacketBuffer(settings.capture_packet_buffer_s)
        self.confirmer = FallConfirmer(
            lambda: share(
                PersonTracker(
                    frame_rate=_CONFIRMATION_TRACKER_FPS,
                    new_track_min_score=settings.tracker_new_track_min_score,
                    track_lost_s=settings.tracker_lost_track_timeout_s,
                )
            ),
            lambda: FallAnalyzer(
                args.camera_id, zones=zones, threshold=settings.alert_min_confidence
            ),
            max_width_px=_NO_RESIZE_PX,  # offline analysis runs at the file's size
            rotate_deg=args.rotate_deg,
            max_fps=settings.fall_confirmation_max_fps,
        )

    def keep(self, record: FrameRecord, packet: Any) -> None:
        """Keep the encoded packet for confirmation, timed like the frames (media time)."""
        if self.pending is None:
            return
        if not self.live and record.pts_s is not None:
            record = dataclasses.replace(record, arrival_ns=int(record.pts_s * 1e9))
        self.packets.add(record, packet)

    def _confirm(self, candidates: list[Event]) -> None:
        for event in candidates:
            start_ns = int((event.start_s - self.before_s) * 1e9)
            end_ns = int((event.end_s + self.after_s) * 1e9)
            for confirmed in self.confirmer.confirm(self.packets.decode_window(start_ns, end_ns)):
                self._write_event(confirmed)

    def frame(self, frame: av.VideoFrame, packet_record: FrameRecord) -> None:
        image = rotate(
            np.asarray(frame.to_ndarray(format="bgr24"), dtype=np.uint8), self.rotate_deg
        )
        time_base = Fraction(str(frame.time_base)) if frame.time_base else packet_record.time_base
        record = dataclasses.replace(
            packet_record,
            seq=self.frames,
            pts=frame.pts,
            time_base=time_base,
            keyframe=bool(frame.key_frame),
            quality=self.classifier.classify(frame.pts, time_base),
        )
        if not self.live and record.pts_s is not None:
            # Files are read faster than real time: schedule on media time.
            record = dataclasses.replace(record, arrival_ns=int(record.pts_s * 1e9))
        result = self.pipeline.process(image, record)
        self.frames += 1
        self.last_s = record.arrival_ns / 1e9
        if result.analysed:
            for event in self.detectors.update(result, (image.shape[1], image.shape[0])):
                self._write_event(event)
                if self.pending is not None:
                    self.pending.add(event)
        if self.pending is not None:
            self._confirm(self.pending.due(self.last_s))
        if result.analysed:
            self.analysed += 1
            self.last_tracks = result.tracks
            entry = {
                "seq": record.seq,
                "arrival_s": round(record.arrival_ns / 1e9, 6),
                "pts_s": record.pts_s,
                "tracks": [_track_json(t) for t in result.tracks],
                "detections": [
                    {
                        "box": [round(v, 1) for v in d.box_xyxy],
                        "score": round(d.score, 3),
                        "track_id": d.track_id,
                    }
                    for d in result.detections
                ],
            }
            self.tracks_file.write(json.dumps(entry) + "\n")
        if self.overlay_path is not None:
            if self.writer is None:
                self.writer = av.open(str(self.overlay_path), "w")
                self.stream = self.writer.add_stream("libx264", rate=25)
                self.stream.width, self.stream.height = image.shape[1], image.shape[0]
                self.stream.pix_fmt = "yuv420p"
            overlay = av.VideoFrame.from_ndarray(draw(image, self.last_tracks), "bgr24")
            for out_packet in self.stream.encode(overlay):
                self.writer.mux(out_packet)

    def _write_event(self, event: Event) -> None:
        """``emitted_s``: the time of the frame at which the event was raised (latency)."""
        self.events += 1
        line = {**_event_json(event), "emitted_s": round(self.last_s, 3)}
        self.events_file.write(json.dumps(line) + "\n")

    def close(self) -> None:
        for event in self.detectors.flush():  # candidates the end of the file left open
            self._write_event(event)
        if self.pending is not None:
            self._confirm(self.pending.remaining())
        if self.writer is not None:
            for out_packet in self.stream.encode():
                self.writer.mux(out_packet)
            self.writer.close()


def run(args: argparse.Namespace) -> int:
    out_dir: Path = args.out_dir or Path("artifacts/analyze") / datetime.now(UTC).strftime(
        "%Y%m%dT%H%M%SZ"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    with (
        (out_dir / "tracks.jsonl").open("w", encoding="utf-8") as tracks_file,
        (out_dir / "events.jsonl").open("w", encoding="utf-8") as events_file,
    ):
        analyzer = _Analyzer(args, out_dir, tracks_file, events_file)
        last: tuple[FrameRecord, Any] | None = None
        try:
            for record, packet in iter_frame_records(args.source, args.camera_id, SourceOptions()):
                last = (record, packet)
                analyzer.keep(record, packet)
                for frame in packet.decode():
                    analyzer.frame(frame, record)
                if args.max_frames and analyzer.frames >= args.max_frames:
                    break
            else:
                if last is not None:
                    # Drain frames still held by the decoder (B-frames).
                    for frame in last[1].stream.codec_context.decode(None):
                        analyzer.frame(frame, last[0])
        finally:
            analyzer.close()
    print(
        f"analyze: {analyzer.frames} frames, {analyzer.analysed} analysed, "
        f"{analyzer.events} events from "
        f"{mask_url(args.source)} -> {out_dir}"
    )
    return 0
