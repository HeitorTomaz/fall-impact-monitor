"""The monitoring service: one worker thread per camera, the outputs, clip export and
maintenance, in one process (``fim run``).

Order of priorities: an event is queued for every output the moment a detector raises
it; nothing on that path waits for the database, the internet or a clip. The database
schema is migrated in the background, retrying until the database answers.
"""

import asyncio
import logging
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any

from sqlalchemy.ext.asyncio import AsyncEngine

from fall_impact_monitor.analysis.audio import ThumpDetector
from fall_impact_monitor.analysis.camera import CameraAnalyzer
from fall_impact_monitor.analysis.fall import FallAnalyzer
from fall_impact_monitor.analysis.zones import load_camera_zones
from fall_impact_monitor.api.confirmation import FallConfirmer
from fall_impact_monitor.capture.audio import AudioReader
from fall_impact_monitor.capture.packet_buffer import LivePacketBuffer
from fall_impact_monitor.capture.reader import ReconnectingReader, SourceOptions, StateEvent
from fall_impact_monitor.config import CameraSettings, Settings
from fall_impact_monitor.config.snapshot import config_hash, config_snapshot
from fall_impact_monitor.events.record import EventRecord
from fall_impact_monitor.perception.tracking import PersonTracker
from fall_impact_monitor.recording.clips import CLIP_WAIT_TIMEOUT_S, EventClipExporter
from fall_impact_monitor.service.camera import (
    Analyzer,
    CameraWorker,
    DecodedFrame,
    Pipeline,
    analysis_url,
    decode,
)
from fall_impact_monitor.service.clips import ClipWorker
from fall_impact_monitor.service.confirmation import ConfirmationWorker
from fall_impact_monitor.service.handover import LatestFrames
from fall_impact_monitor.service.health import HealthLoop, LoopWatchdog
from fall_impact_monitor.service.maintenance import maintenance_loop, maintenance_step
from fall_impact_monitor.sinks.factory import build_dispatcher
from fall_impact_monitor.sinks.telegram.api import TelegramApi
from fall_impact_monitor.sinks.telegram.bot import keep_polling, run_polling
from fall_impact_monitor.sinks.telegram.controller import TelegramController
from fall_impact_monitor.storage import make_engine, upgrade

log = logging.getLogger(__name__)
_HEARTBEAT_S = 10.0
_MAINTENANCE_S = 600.0
_CONFIRMATION_TRACKER_FPS = 25.0  # confirmation analyses every frame of the stream
# A press before the clip exists waits for the post-event window, the exporter's own
# wait for the recording, and the export itself (a re-encode, a queue of clips).
_CLIP_EXPORT_ALLOWANCE_S = 120.0


@dataclass
class ServiceParts:
    """What the service builds per camera; tests replace them with fakes."""

    make_pipeline: Callable[[], Pipeline]
    make_analyzer: Callable[[CameraSettings], Analyzer]
    make_source: Callable[[CameraSettings], Iterable[DecodedFrame | StateEvent]]
    telegram_api: TelegramApi | None = None
    # Audio for thump evidence: (camera, analyzer) -> a reader with run()/stop(), or None.
    make_audio: Callable[[CameraSettings, Analyzer], AudioReader | None] | None = None
    # Encoded packets per camera id, fed by make_source (full-rate confirmation).
    packet_buffers: dict[str, LivePacketBuffer] = field(default_factory=dict)


def default_parts(settings: Settings) -> ServiceParts:
    from fall_impact_monitor.api import build_pipeline

    buffers: dict[str, LivePacketBuffer] = {}

    def analyzer(camera: CameraSettings) -> Analyzer:
        zones = (
            load_camera_zones(
                Path(camera.zones_file),
                rotate_deg=camera.rotate_deg,
                variable=f"FIM_CAMERA_{camera.id.upper()}_ROTATE_DEG",
            )
            if camera.zones_file
            else []
        )
        return CameraAnalyzer(
            camera.id,
            zones,
            threshold=settings.alert_min_confidence,
            fall_enabled=settings.fall_detector_enabled,
            floor_enabled=settings.person_down_detector_enabled,
            person_down_s=settings.person_down_detector_after_fall_s,
            no_fall_s=settings.person_down_detector_without_fall_s,
            head_impact_enabled=settings.head_impact_detector_enabled,
            head_impact_min_evidence=settings.head_impact_min_evidence,
            head_impact_window_s=settings.head_impact_window_s,
        )

    def audio(camera: CameraSettings, analyzer: Analyzer) -> AudioReader | None:
        add_thumps = getattr(analyzer, "add_thumps", None)
        wants_audio = getattr(analyzer, "wants_audio", None)
        if (
            not settings.audio_enabled
            or not callable(add_thumps)
            or not (callable(wants_audio) and wants_audio())
        ):
            return None
        detector = ThumpDetector(min_db=settings.audio_thump_min_db)
        # The main stream path carries the camera's audio (the substream may not).
        url = f"{settings.mediamtx_rtsp_url.rstrip('/')}/{camera.id}"
        return AudioReader(url, camera.id, lambda t, s: add_thumps(detector.feed(t, s)))

    def source(camera: CameraSettings) -> Iterable[DecodedFrame | StateEvent]:
        reader = ReconnectingReader(
            analysis_url(settings.mediamtx_rtsp_url, camera),
            camera.id,
            SourceOptions(),
            stale_timeout_s=settings.capture_stale_timeout_s,
        )
        # Reading runs in its own thread so that slow analysis skips frames instead of
        # falling behind real time (see handover.py).
        buffer = None
        if settings.fall_confirmation_enabled:
            buffer = buffers.setdefault(
                camera.id, LivePacketBuffer(settings.capture_packet_buffer_s)
            )
        frames = decode(reader, rotate_deg=camera.rotate_deg, buffer=buffer)
        return LatestFrames(frames, camera.id, on_close=reader.close)

    return ServiceParts(
        make_pipeline=lambda: build_pipeline(settings, started_s=time.monotonic()),
        make_analyzer=analyzer,
        make_source=source,
        make_audio=audio,
        packet_buffers=buffers,
    )


def confirmation_worker(
    settings: Settings,
    cameras: list[CameraSettings],
    pipelines: dict[str, Pipeline],
    buffers: dict[str, LivePacketBuffer],
    on_event: Callable[[EventRecord], None],
) -> ConfirmationWorker | None:
    """Full-rate confirmation for the cameras whose pipeline can share its models."""
    if not settings.fall_confirmation_enabled:
        return None
    by_id = {camera.id: camera for camera in cameras}
    zones = {
        camera.id: load_camera_zones(
            Path(camera.zones_file),
            rotate_deg=camera.rotate_deg,
            variable=f"FIM_CAMERA_{camera.id.upper()}_ROTATE_DEG",
        )
        if camera.zones_file
        else []
        for camera in cameras
    }
    shares: dict[str, Callable[[PersonTracker], Any]] = {
        camera_id: share
        for camera_id in buffers
        if callable(share := getattr(pipelines.get(camera_id), "sharing_models", None))
    }
    usable = {camera_id: buffers[camera_id] for camera_id in shares}
    if not usable:
        return None

    def confirmer_for(camera_id: str) -> FallConfirmer:
        share = shares[camera_id]
        camera = by_id[camera_id]
        return FallConfirmer(
            lambda: share(
                PersonTracker(
                    frame_rate=_CONFIRMATION_TRACKER_FPS,
                    new_track_min_score=settings.tracker_new_track_min_score,
                    track_lost_s=settings.tracker_lost_track_timeout_s,
                )
            ),
            lambda: FallAnalyzer(
                camera_id, zones=zones[camera_id], threshold=settings.alert_min_confidence
            ),
            max_width_px=settings.capture_max_analysis_width_px,
            rotate_deg=camera.rotate_deg,
            max_fps=settings.fall_confirmation_max_fps,
        )

    return ConfirmationWorker(
        confirmer_for,
        usable,
        on_event,
        before_s=settings.fall_confirmation_before_s,
        after_s=settings.fall_confirmation_after_s,
    )


async def _migrate(url: str) -> None:
    delay_s = 5.0
    while True:
        try:
            await asyncio.to_thread(upgrade, url)
            log.info("database schema up to date")
            return
        except Exception as error:
            log.warning("database not ready", extra={"error": type(error).__name__})
            await asyncio.sleep(delay_s)
            delay_s = min(delay_s * 2, 300.0)


async def _heartbeat(path: Path) -> None:
    while True:
        path.write_text(f"{time.time():.0f}\n", encoding="utf-8")
        await asyncio.sleep(_HEARTBEAT_S)


async def run_service(
    settings: Settings,
    cameras: list[CameraSettings],
    *,
    parts: ServiceParts | None = None,
    heartbeat: Path = Path("/tmp/fim-alive"),
    stop: asyncio.Event | None = None,
) -> None:
    parts = parts or default_parts(settings)
    stop = stop or asyncio.Event()
    snapshot = config_snapshot(settings, cameras)
    digest = config_hash(snapshot)
    engine: AsyncEngine | None = (
        make_engine(settings.database_url()) if settings.db_password.get_secret_value() else None
    )
    dispatcher, outputs = build_dispatcher(settings, engine=engine, telegram_api=parts.telegram_api)
    dispatcher.submit_item({"kind": "config", "config_hash": digest, "snapshot": snapshot})
    exporter = EventClipExporter(
        settings.mediamtx_playback_url,
        Path(settings.event_clip_dir),
        pre_s=settings.event_clip_pre_s,
        post_s=settings.event_clip_post_s,
        max_bytes=int(settings.event_clip_max_mb * 1_000_000),
    )
    clips = ClipWorker(exporter.export)

    confirmations: ConfirmationWorker | None = None

    def on_event(record: EventRecord) -> None:
        dispatcher.submit(record)
        clips.request(record)
        if confirmations is not None:
            confirmations.request(record)  # a fall near-miss: analyse it again, every frame

    analyzers = [parts.make_analyzer(camera) for camera in cameras]
    pipelines = {camera.id: parts.make_pipeline() for camera in cameras}
    confirmations = confirmation_worker(
        settings, cameras, pipelines, parts.packet_buffers, on_event
    )
    workers = [
        CameraWorker(
            camera.id,
            pipeline=pipelines[camera.id],
            analyzer=analyzer,
            on_event=on_event,
            config_hash=digest,
            max_width_px=settings.capture_max_analysis_width_px,
            reconnect_window_s=settings.health_reconnect_window_s,
            rotate_deg=camera.rotate_deg,
        )
        for camera, analyzer in zip(cameras, analyzers, strict=True)
    ]
    audio_readers = [
        reader
        for camera, analyzer in zip(cameras, analyzers, strict=True)
        if parts.make_audio is not None
        and (reader := parts.make_audio(camera, analyzer)) is not None
    ]

    health = HealthLoop(workers, dispatcher, settings=settings, config_hash=digest)
    loop_watchdog = LoopWatchdog(stall_s=settings.health_stall_exit_s)

    def status_text() -> str:
        pending = sum(dispatcher.pending_counts().values())
        return f"{health.status_text()}\npending deliveries: {pending}"

    tasks = [
        asyncio.create_task(dispatcher.run()),
        asyncio.create_task(_heartbeat(heartbeat)),
        asyncio.create_task(health.run(loop_watchdog)),
        asyncio.create_task(
            maintenance_loop(
                partial(maintenance_step, settings, cameras=cameras, engine=engine),
                interval_s=_MAINTENANCE_S,
            )
        ),
    ]
    if engine is not None:
        tasks.append(asyncio.create_task(_migrate(settings.database_url())))
    if outputs.bot is not None and outputs.telegram_api is not None:
        controller = TelegramController(
            outputs.telegram_api,
            outputs.telegram,
            dispatcher.submit_item,
            clip_dir=Path(settings.event_clip_dir),
            language=settings.alert_language,
            timezone=settings.alert_timezone,
            status_text=status_text,
            clip_wait_s=settings.event_clip_post_s + CLIP_WAIT_TIMEOUT_S + _CLIP_EXPORT_ALLOWANCE_S,
        )
        bot = outputs.bot
        tasks.append(asyncio.create_task(keep_polling(partial(run_polling, bot, controller))))
    clips.start()
    if confirmations is not None:
        confirmations.start()
    loop_watchdog.start()
    threads = [
        threading.Thread(
            target=worker.run_forever,
            args=(partial(parts.make_source, camera),),
            name=camera.id,
            daemon=True,
        )
        for worker, camera in zip(workers, cameras, strict=True)
    ]
    for thread in threads:
        thread.start()
    for reader in audio_readers:
        threading.Thread(target=reader.run, name="audio", daemon=True).start()
    log.info("monitoring started", extra={"cameras": [c.id for c in cameras], "config": digest})
    try:
        await stop.wait()
    finally:
        for worker in workers:
            worker.stop()
        for reader in audio_readers:
            reader.stop()
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        clips.stop()
        if confirmations is not None:
            confirmations.stop()
        if outputs.bot is not None:
            await outputs.bot.session.close()
        if engine is not None:
            await engine.dispose()
        log.info("monitoring stopped")
