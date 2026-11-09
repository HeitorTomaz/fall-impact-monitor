import os
import time
from datetime import UTC, datetime
from pathlib import Path

from fall_impact_monitor.config import Settings
from fall_impact_monitor.events import EventCategory
from fall_impact_monitor.events.wording import alert_text
from fall_impact_monitor.health.monitor import Condition, Transition
from fall_impact_monitor.health.observe import CameraProbe, health_record, observe
from fall_impact_monitor.sinks.outbox import Outbox

NOW = 1000.0


def conditions(**kw: object) -> set[Condition]:
    args: dict[str, object] = {
        "cameras": [CameraProbe("cam1", last_frame_s=NOW - 1, started_s=0.0, rate_degraded=False)],
        "camera_lost_s": 30.0,
        "disk_free_bytes": 100,
        "min_free_bytes": 10,
        "backlog_ages_s": {},
        "backlog_max_age_s": 300.0,
    }
    args.update(kw)
    return observe(NOW, **args)  # type: ignore[arg-type]


def test_a_healthy_system_has_no_conditions() -> None:
    assert conditions() == set()


def test_a_camera_without_video_is_lost() -> None:
    probe = CameraProbe("cam1", last_frame_s=NOW - 31, started_s=0.0, rate_degraded=True)
    assert conditions(cameras=[probe]) == {Condition("cam1", "camera_lost")}


def test_a_camera_that_never_connected_is_lost_after_the_limit() -> None:
    young = CameraProbe("cam1", last_frame_s=None, started_s=NOW - 10, rate_degraded=False)
    old = CameraProbe("cam1", last_frame_s=None, started_s=NOW - 40, rate_degraded=False)
    assert conditions(cameras=[young]) == set()
    assert conditions(cameras=[old]) == {Condition("cam1", "camera_lost")}


def test_slow_analysis_is_reported_while_video_arrives() -> None:
    probe = CameraProbe("cam1", last_frame_s=NOW - 1, started_s=0.0, rate_degraded=True)
    assert conditions(cameras=[probe]) == {Condition("cam1", "analysis_slow")}


def test_disk_and_delivery_backlogs_degrade_the_system() -> None:
    found = conditions(
        disk_free_bytes=5,
        backlog_ages_s={
            "telegram:ab12": 400.0,
            "webhook": 10.0,
            "database": 900.0,
            "event_log": 999.0,
        },
    )
    assert found == {
        Condition("system", "disk_low"),
        Condition("system", "alerts_backlog"),
        Condition("system", "database_backlog"),
    }


def test_the_outbox_reports_the_age_of_its_oldest_item(tmp_path: Path) -> None:
    outbox = Outbox(tmp_path)
    assert outbox.oldest_age_s("telegram:x") is None
    outbox.put("telegram:x", {"kind": "test"})
    path = next((tmp_path / "telegram_x").glob("*.json"))
    os.utime(path, (time.time() - 120, time.time() - 120))
    age = outbox.oldest_age_s("telegram:x")
    assert age is not None and 119 <= age <= 125


def test_transitions_become_localized_monitoring_messages() -> None:
    wall = datetime(2026, 11, 9, 3, 0, tzinfo=UTC)
    record = health_record(
        Transition("cam1", degraded=True, reasons=("camera_lost",)),
        wall_utc=wall,
        config_hash="cfg",
    )
    assert record.event.category is EventCategory.MONITORING_DEGRADED
    assert record.event.camera_id == "cam1" and not record.event.near_miss
    text = alert_text(record, language="pt-BR", timezone="UTC", now_utc=wall, late_after_s=60)
    assert "Monitoramento degradado" in text and "sem vídeo da câmera" in text
    restored = health_record(
        Transition("cam1", degraded=False, reasons=()), wall_utc=wall, config_hash="cfg"
    )
    assert restored.event.category is EventCategory.MONITORING_RESTORED


def test_health_settings_defaults() -> None:
    s = Settings(_env_file=None)
    assert s.health_camera_lost_s == 30.0 and s.health_camera_lost_s <= 60
    assert (s.health_check_interval_s, s.health_restore_after_s) == (5.0, 30.0)
    assert (s.health_backlog_max_age_s, s.health_stall_exit_s) == (300.0, 180.0)


def test_a_camera_that_just_stopped_is_not_reported_as_slow_analysis() -> None:
    # Frames stopped 9 s ago: the rate falls too, but the cause is the missing video,
    # reported as camera_lost once the limit passes.
    probe = CameraProbe("cam1", last_frame_s=NOW - 9, started_s=0.0, rate_degraded=True)
    assert conditions(cameras=[probe]) == set()
