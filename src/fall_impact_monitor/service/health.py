"""The health loop: observes cameras, disk and deliveries every few seconds, reports
degraded / restored monitoring through the dispatcher, and exits the process when a
camera worker is stuck so Docker restarts the container (watchdog).

A camera that is merely offline is not stuck: its reader keeps reporting stream states
while it reconnects. Only a worker with no activity at all (no frame and no state) for
``FIM_HEALTH_STALL_EXIT_S`` triggers the watchdog.
"""

import asyncio
import json
import logging
import os
import shutil
import tempfile
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from fall_impact_monitor.config import Settings
from fall_impact_monitor.events.record import EventRecord
from fall_impact_monitor.health.monitor import HealthMonitor, Transition
from fall_impact_monitor.health.observe import SYSTEM, HealthLimits, health_record, observe
from fall_impact_monitor.service.camera import CameraWorker

log = logging.getLogger(__name__)
STALL_EXIT_CODE = 70
_GB = 1_000_000_000


class _Dispatcher(Protocol):
    def submit(self, record: EventRecord) -> None: ...

    def oldest_pending_age_s(self) -> dict[str, float | None]: ...


def _disk_free(path: Path) -> int:
    target = path if path.exists() else Path("/")
    return shutil.disk_usage(target).free


class HealthLoop:
    def __init__(
        self,
        workers: list[CameraWorker],
        dispatcher: _Dispatcher,
        *,
        settings: Settings,
        config_hash: str,
        monotonic: Callable[[], float] = time.monotonic,
        wall: Callable[[], datetime] = lambda: datetime.now(UTC),
        disk_free_bytes: Callable[[], int] | None = None,
        exit: Callable[[int], object] = os._exit,
    ) -> None:
        self._workers = workers
        self._dispatcher = dispatcher
        self._settings = settings
        self._config_hash = config_hash
        self._monotonic = monotonic
        self._wall = wall
        self._disk_free_bytes = disk_free_bytes or (
            lambda: _disk_free(Path(settings.recordings_dir))
        )
        self._exit = exit
        self._limits = HealthLimits(
            camera_lost_s=settings.health_camera_lost_s,
            camera_gap_s=settings.capture_stale_timeout_s,
            max_reconnects=settings.health_camera_max_reconnects,
            reconnect_window_s=settings.health_reconnect_window_s,
            max_processing_failures=settings.health_max_processing_failures,
            min_free_bytes=int(settings.recordings_disk_min_free_gb * _GB),
            alerts_max_age_s=settings.alert_late_after_s,
            database_max_age_s=settings.health_database_backlog_max_age_s,
        )
        self._monitor = HealthMonitor(
            [w.camera_id for w in workers] + [SYSTEM],
            restore_after_s=settings.health_restore_after_s,
        )
        self._state_file = Path(settings.health_state_file)
        self._monitor.restore(self._load_state(), self._monotonic())

    def _load_state(self) -> dict[str, list[str]]:
        """What was announced before a restart; nothing when the file is missing or bad."""
        try:
            data = json.loads(self._state_file.read_text(encoding="utf-8"))
            return {
                str(name): [str(r) for r in reasons]
                for name, reasons in data["announced"].items()
                if isinstance(reasons, list)
            }
        except FileNotFoundError:
            return {}
        except (OSError, ValueError, KeyError, AttributeError, TypeError) as error:
            log.warning("health state not loaded", extra={"error": type(error).__name__})
            return {}

    def _save_state(self) -> None:
        """Atomically, so a crash never leaves half a file; a failure is only logged."""
        try:
            folder = self._state_file.parent
            folder.mkdir(parents=True, exist_ok=True)
            handle, name = tempfile.mkstemp(prefix=".health-", suffix=".tmp", dir=folder)
            try:
                with os.fdopen(handle, "w", encoding="utf-8") as out:
                    json.dump({"announced": self._monitor.snapshot()}, out)
                os.replace(name, self._state_file)
            finally:
                Path(name).unlink(missing_ok=True)
        except OSError as error:
            log.warning("health state not saved", extra={"error": type(error).__name__})

    def _report(self, transition: Transition) -> None:
        log.warning(
            "monitoring degraded" if transition.degraded else "monitoring restored",
            extra={"component": transition.component, "reasons": list(transition.reasons)},
        )
        record = health_record(transition, wall_utc=self._wall(), config_hash=self._config_hash)
        self._dispatcher.submit(record)

    def check(self) -> None:
        s = self._settings
        now_s = self._monotonic()
        for worker in self._workers:
            if now_s - worker.last_activity_s > s.health_stall_exit_s:
                log.critical("camera worker stuck, exiting", extra={"camera": worker.camera_id})
                try:
                    self._report(Transition(worker.camera_id, True, ("worker_stalled",)))
                except Exception:
                    log.exception("could not report the stall")
                self._exit(STALL_EXIT_CODE)  # always, even if reporting failed
                return
        degraded = {
            name for name, (state, _) in self._monitor.states().items() if state == "DEGRADED"
        }
        conditions = observe(
            now_s,
            cameras=[w.probe(now_s) for w in self._workers],
            limits=self._limits,
            degraded=degraded,
            disk_free_bytes=self._disk_free_bytes(),
            backlog_ages_s=self._dispatcher.oldest_pending_age_s(),
        )
        transitions = self._monitor.update(now_s, conditions)
        for transition in transitions:
            self._report(transition)
        if transitions:
            self._save_state()

    def status_text(self) -> str:
        lines = []
        for component, (state, reasons) in self._monitor.states().items():
            lines.append(f"{component}: {state}" + (f" ({', '.join(reasons)})" if reasons else ""))
        lines += [worker.status_line() for worker in self._workers]
        return "\n".join(lines)

    async def run(self, watchdog: "LoopWatchdog | None" = None) -> None:
        while True:
            if watchdog is not None:
                watchdog.tick()
            try:
                self.check()
            except Exception:
                log.exception("health check failed")
            await asyncio.sleep(self._settings.health_check_interval_s)


class LoopWatchdog:
    """Exits the process when the asyncio loop stops ticking (a hung disk, a blocking
    call): the health loop runs inside that loop and could not notice by itself."""

    def __init__(
        self,
        *,
        stall_s: float,
        monotonic: Callable[[], float] = time.monotonic,
        exit: Callable[[int], object] = os._exit,
    ) -> None:
        self._stall_s = stall_s
        self._monotonic = monotonic
        self._exit = exit
        self._last_tick_s = monotonic()

    def tick(self) -> None:
        self._last_tick_s = self._monotonic()

    def check(self) -> None:
        if self._monotonic() - self._last_tick_s > self._stall_s:
            log.critical("event loop stuck, exiting")
            self._exit(STALL_EXIT_CODE)

    def start(self, interval_s: float = 5.0) -> None:
        def loop() -> None:
            while True:
                time.sleep(interval_s)
                self.check()

        threading.Thread(target=loop, name="loop-watchdog", daemon=True).start()
