"""The health loop: observes cameras, disk and deliveries every few seconds, reports
degraded / restored monitoring through the dispatcher, and exits the process when a
camera worker is stuck so Docker restarts the container (watchdog).

A camera that is merely offline is not stuck: its reader keeps reporting stream states
while it reconnects. Only a worker with no activity at all (no frame and no state) for
``FIM_HEALTH_STALL_EXIT_S`` triggers the watchdog.
"""

import asyncio
import logging
import os
import shutil
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from fall_impact_monitor.config import Settings
from fall_impact_monitor.events.record import EventRecord
from fall_impact_monitor.health.monitor import HealthMonitor, Transition
from fall_impact_monitor.health.observe import SYSTEM, health_record, observe
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
        self._monitor = HealthMonitor(
            [w.camera_id for w in workers] + [SYSTEM],
            restore_after_s=settings.health_restore_after_s,
        )

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
                self._report(Transition(worker.camera_id, True, ("worker_stalled",)))
                self._exit(STALL_EXIT_CODE)
                return
        conditions = observe(
            now_s,
            cameras=[w.probe(now_s) for w in self._workers],
            camera_lost_s=s.health_camera_lost_s,
            disk_free_bytes=self._disk_free_bytes(),
            min_free_bytes=int(s.recordings_disk_min_free_gb * _GB),
            backlog_ages_s=self._dispatcher.oldest_pending_age_s(),
            backlog_max_age_s=s.health_backlog_max_age_s,
        )
        for transition in self._monitor.update(now_s, conditions):
            self._report(transition)

    def status_text(self) -> str:
        lines = []
        for component, (state, reasons) in self._monitor.states().items():
            lines.append(f"{component}: {state}" + (f" ({', '.join(reasons)})" if reasons else ""))
        lines += [worker.status_line() for worker in self._workers]
        return "\n".join(lines)

    async def run(self) -> None:
        while True:
            try:
                self.check()
            except Exception:
                log.exception("health check failed")
            await asyncio.sleep(self._settings.health_check_interval_s)
