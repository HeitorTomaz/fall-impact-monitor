"""`fim analyze`: run perception and the detectors on a file or stream; write tracks, events
and an optional overlay.

``tracks.jsonl`` has one line per frame read (perception results at full precision for
analysed frames) and ``confirmations.jsonl`` the perception of every full-rate
re-analysis window, so that ``fim eval --replay`` can run the detectors again exactly as
here without perception (#67).

Outputs go to `artifacts/` for local inspection only; never publish them.
"""

import argparse
import dataclasses
import json
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
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
from fall_impact_monitor.analysis.zones import Zone, load_zones
from fall_impact_monitor.api import Track, build_pipeline
from fall_impact_monitor.api.confirmation import FallConfirmer, PendingConfirmations, decide
from fall_impact_monitor.api.perception import DetectedPerson, FrameResult
from fall_impact_monitor.capture.packet_buffer import LivePacketBuffer
from fall_impact_monitor.capture.reader import SourceOptions, is_network_source, iter_frame_records
from fall_impact_monitor.capture.rotation import ROTATIONS_DEG, rotate
from fall_impact_monitor.capture.timing import (
    FrameRecord,
    TimestampClassifier,
    TimestampQuality,
)
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
ANALYSIS_FORMAT = 2  # layout of tracks.jsonl and confirmations.jsonl; replay needs it equal
_NO_RESIZE_PX = 1_000_000


def _positive(text: str) -> float:
    value = float(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return value


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
    parser.add_argument(
        "--media-time-scale",
        type=_positive,
        default=1.0,
        help="files only: multiply media time by this, for a file whose declared frame "
        "rate is wrong (4 for a file declaring 120 fps that was captured at 30)",
    )
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
    """Full precision: a replay must see the values the detectors saw."""
    return {
        "id": track.track_id,
        "box": list(track.box_xyxy),
        "score": track.score,
        "identity": track.identity,
        "keypoints": None if track.keypoints is None else track.keypoints.tolist(),
        "keypoint_scores": None
        if track.keypoint_scores is None
        else track.keypoint_scores.tolist(),
        "age_frames": track.age_frames,
        "detection_index": track.detection_index,
    }


def frame_entry(result: FrameResult, size: tuple[int, int]) -> dict[str, Any]:
    """One analysed frame of ``tracks.jsonl`` (``arrival_s`` and ``pts_s`` for reading)."""
    record = result.record
    return {
        "seq": record.seq,
        "arrival_ns": record.arrival_ns,
        "arrival_s": round(record.arrival_ns / 1e9, 6),
        "pts_s": record.pts_s,
        "quality": record.quality.value,
        "analysed": True,
        "size": list(size),
        "tracks": [_track_json(t) for t in result.tracks],
        "detections": [
            {"box": list(d.box_xyxy), "score": d.score, "track_id": d.track_id}
            for d in result.detections
        ],
    }


def _skipped_entry(record: FrameRecord) -> dict[str, Any]:
    """A frame read but not analysed: its time still drives the re-analysis."""
    return {"seq": record.seq, "arrival_ns": record.arrival_ns, "analysed": False}


def _saved_record(entry: dict[str, Any]) -> FrameRecord:
    return FrameRecord(
        "replay",
        entry["seq"],
        None,
        Fraction(1),
        entry["arrival_ns"],
        datetime.fromtimestamp(0, UTC),
        False,
        0,
        TimestampQuality(entry.get("quality", TimestampQuality.OK.value)),
    )


def _array(values: list[Any] | None) -> NDArray[np.float32] | None:
    return None if values is None else np.array(values, np.float32)


def _box(values: list[float]) -> tuple[float, float, float, float]:
    return (values[0], values[1], values[2], values[3])


def result_from_entry(entry: dict[str, Any]) -> tuple[FrameResult, tuple[int, int]]:
    """An analysed frame read back from ``tracks.jsonl``, with its frame size."""
    tracks = [
        Track(
            t["id"],
            _box(t["box"]),
            t["score"],
            _array(t["keypoints"]),
            _array(t["keypoint_scores"]),
            t["identity"],
            t["age_frames"],
            t["detection_index"],
        )
        for t in entry["tracks"]
    ]
    detections = [
        DetectedPerson(_box(d["box"]), d["score"], None, d["track_id"]) for d in entry["detections"]
    ]
    size = (entry["size"][0], entry["size"][1])
    return FrameResult(_saved_record(entry), tracks, analysed=True, detections=detections), size


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


def _camera_analyzer(settings: Settings, camera_id: str, zones: list[Zone]) -> CameraAnalyzer:
    return CameraAnalyzer(
        camera_id,
        zones,
        threshold=settings.alert_min_confidence,
        fall_enabled=settings.fall_detector_enabled,
        floor_enabled=settings.person_down_detector_enabled,
        person_down_s=settings.person_down_detector_after_fall_s,
        no_fall_s=settings.person_down_detector_without_fall_s,
    )


def _fall_rule(settings: Settings, camera_id: str, zones: list[Zone]) -> Callable[[], FallAnalyzer]:
    return lambda: FallAnalyzer(camera_id, zones=zones, threshold=settings.alert_min_confidence)


ConfirmWindow = Callable[[int, int], list[Event]]  # (start_ns, end_ns) -> alerts


class RuleRunner:
    """Everything after perception, frame by frame: the detectors, the full-rate
    re-analysis of fall near-misses once their window is complete, and the events.
    Analysis and replay share it, so a replay runs the same steps."""

    def __init__(
        self,
        settings: Settings,
        detectors: CameraAnalyzer,
        write: Callable[[Event, float], None],
        confirm_window: ConfirmWindow | None,
    ) -> None:
        self._detectors = detectors
        self._write = write
        self._confirm_window = confirm_window
        self._before_s = settings.fall_confirmation_before_s
        self._after_s = settings.fall_confirmation_after_s
        self._pending = PendingConfirmations(self._after_s) if confirm_window else None
        self.last_s = 0.0  # time of the last frame read

    def frame(self, result: FrameResult, size: tuple[int, int]) -> None:
        self.last_s = result.record.arrival_ns / 1e9
        if result.analysed:
            for event in self._detectors.update(result, size):
                self._write(event, self.last_s)
                if self._pending is not None:
                    self._pending.add(event)
        if self._pending is not None:
            self._confirm(self._pending.due(self.last_s))

    def close(self) -> None:
        for event in self._detectors.flush():  # candidates the end of the file left open
            self._write(event, self.last_s)
        if self._pending is not None:
            self._confirm(self._pending.remaining())

    def _confirm(self, candidates: list[Event]) -> None:
        if self._confirm_window is None:
            return
        for event in candidates:
            start_ns = int((event.start_s - self._before_s) * 1e9)
            end_ns = int((event.end_s + self._after_s) * 1e9)
            for confirmed in self._confirm_window(start_ns, end_ns):
                self._write(confirmed, self.last_s)


@dataclass(frozen=True)
class ReplayResult:
    events: list[dict[str, Any]]  # as in events.jsonl
    missing_windows: int  # candidates whose re-analysis window was not saved


def _saved_lines(path: Path) -> Iterator[dict[str, Any]]:
    if path.exists():
        with path.open(encoding="utf-8") as lines:
            for line in lines:
                if line.strip():
                    yield json.loads(line)


def replay(out_dir: Path, settings: Settings, camera_id: str = "analyze") -> ReplayResult:
    """The detectors of ``settings`` run again on a saved analysis, without perception.

    A near-miss whose re-analysis window was not saved (the rules changed which
    candidates exist) is counted in ``missing_windows``; its re-analysis needs a full run.
    """
    windows = {
        (w["start_ns"], w["end_ns"]): [result_from_entry(f) for f in w["frames"]]
        for w in _saved_lines(out_dir / "confirmations.jsonl")
    }
    events: list[dict[str, Any]] = []
    missing = 0
    make_rule = _fall_rule(settings, camera_id, [])

    def saved_window(start_ns: int, end_ns: int) -> list[Event]:
        nonlocal missing
        frames = windows.get((start_ns, end_ns))
        if frames is None:
            missing += 1
            return []
        return decide(make_rule, frames)

    def write(event: Event, emitted_s: float) -> None:
        events.append({**_event_json(event), "emitted_s": round(emitted_s, 3)})

    runner = RuleRunner(
        settings,
        _camera_analyzer(settings, camera_id, []),
        write,
        saved_window if settings.fall_confirmation_enabled else None,
    )
    size = (0, 0)
    for entry in _saved_lines(out_dir / "tracks.jsonl"):
        if entry["analysed"]:
            result, size = result_from_entry(entry)
        else:
            result = FrameResult(_saved_record(entry), [], analysed=False)
        runner.frame(result, size)
    runner.close()
    return ReplayResult(events, missing)


class _Analyzer:
    """Feeds decoded frames, each described by its own presentation timestamp, to the
    pipeline and writes tracks.jsonl, confirmations.jsonl and the optional overlay."""

    def __init__(self, args: argparse.Namespace, out_dir: Path, files: dict[str, Any]) -> None:
        settings = Settings()
        self.pipeline = build_pipeline(
            settings, started_s=time.monotonic(), manifest_path=args.manifest
        )
        zones = load_zones(args.zones) if args.zones else []
        self.events_file = files["events"]
        self.tracks_file = files["tracks"]
        self.confirmations_file = files["confirmations"]
        self.events = 0
        self.live = is_network_source(args.source)
        self.time_scale: float = args.media_time_scale
        self.rotate_deg: int = args.rotate_deg
        self.overlay_path = out_dir / "overlay.mp4" if args.overlay else None
        self.classifier = TimestampClassifier()
        self.writer: Any = None
        self.stream: Any = None
        self.last_tracks: list[Track] = []
        self.frames = 0
        self.analysed = 0
        self.confirmer = self._confirmer(settings, args, zones)
        self.packets = LivePacketBuffer(settings.capture_packet_buffer_s)
        self.rules = RuleRunner(
            settings,
            _camera_analyzer(settings, args.camera_id, zones),
            self._write_event,
            self._confirm_window if self.confirmer is not None else None,
        )

    def _confirmer(
        self, settings: Settings, args: argparse.Namespace, zones: list[Zone]
    ) -> FallConfirmer | None:
        """Fall near-misses are analysed again at full rate, as in the service."""
        share = getattr(self.pipeline, "sharing_models", None)
        if not settings.fall_confirmation_enabled or not callable(share):
            return None
        return FallConfirmer(
            lambda: share(
                PersonTracker(
                    frame_rate=_CONFIRMATION_TRACKER_FPS,
                    new_track_min_score=settings.tracker_new_track_min_score,
                    track_lost_s=settings.tracker_lost_track_timeout_s,
                )
            ),
            _fall_rule(settings, args.camera_id, zones),
            max_width_px=_NO_RESIZE_PX,  # offline analysis runs at the file's size
            rotate_deg=args.rotate_deg,
            max_fps=settings.fall_confirmation_max_fps,
        )

    def keep(self, record: FrameRecord, packet: Any) -> None:
        """Keep the encoded packet for confirmation, timed like the frames (media time)."""
        if self.confirmer is None:
            return
        self.packets.add(self._media_timed(record), packet)

    def _media_timed(self, record: FrameRecord) -> FrameRecord:
        """Files are read faster than real time: schedule on (scaled) media time."""
        if self.live or record.pts_s is None:
            return record
        return dataclasses.replace(record, arrival_ns=int(record.pts_s * self.time_scale * 1e9))

    def _confirm_window(self, start_ns: int, end_ns: int) -> list[Event]:
        """Re-analyse the window, and save its perception for a replay."""
        assert self.confirmer is not None
        perceived = list(self.confirmer.perceive(self.packets.decode_window(start_ns, end_ns)))
        saved = {
            "start_ns": start_ns,
            "end_ns": end_ns,
            "frames": [frame_entry(result, size) for result, size in perceived],
        }
        self.confirmations_file.write(json.dumps(saved) + "\n")
        return self.confirmer.decide(perceived)

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
        record = self._media_timed(record)
        result = self.pipeline.process(image, record)
        self.frames += 1
        size = (image.shape[1], image.shape[0])
        self.rules.frame(result, size)
        if result.analysed:
            self.analysed += 1
            self.last_tracks = result.tracks
            self.tracks_file.write(json.dumps(frame_entry(result, size)) + "\n")
        else:
            self.tracks_file.write(json.dumps(_skipped_entry(record)) + "\n")
        if self.overlay_path is not None:
            if self.writer is None:
                self.writer = av.open(str(self.overlay_path), "w")
                self.stream = self.writer.add_stream("libx264", rate=25)
                self.stream.width, self.stream.height = image.shape[1], image.shape[0]
                self.stream.pix_fmt = "yuv420p"
            overlay = av.VideoFrame.from_ndarray(draw(image, self.last_tracks), "bgr24")
            for out_packet in self.stream.encode(overlay):
                self.writer.mux(out_packet)

    def _write_event(self, event: Event, emitted_s: float) -> None:
        """``emitted_s``: the time of the frame at which the event was raised (latency)."""
        self.events += 1
        line = {**_event_json(event), "emitted_s": round(emitted_s, 3)}
        self.events_file.write(json.dumps(line) + "\n")

    def close(self) -> None:
        self.rules.close()
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
        (out_dir / "confirmations.jsonl").open("w", encoding="utf-8") as confirmations_file,
    ):
        files = {"tracks": tracks_file, "events": events_file, "confirmations": confirmations_file}
        analyzer = _Analyzer(args, out_dir, files)
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
