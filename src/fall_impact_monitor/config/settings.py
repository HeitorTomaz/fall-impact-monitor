"""Application settings read from ``FIM_*`` environment variables."""

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, Self
from urllib.parse import quote
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_CAMERA_ID = re.compile(r"[a-z0-9_]+")


class Settings(BaseSettings):
    """Process-wide settings. Every field maps to ``FIM_<FIELD_NAME>``."""

    model_config = SettingsConfigDict(
        env_prefix="FIM_", env_file=".env", extra="ignore", hide_input_in_errors=True
    )

    log_level: str = "INFO"
    capture_stale_timeout_s: float = 10.0
    capture_packet_buffer_s: float = 20.0
    capture_max_analysis_width_px: int = Field(default=1280, ge=160)
    mediamtx_rtsp_url: str = "rtsp://mediamtx:8554"
    mediamtx_playback_url: str = "http://mediamtx:9996"
    models_dir: str = "models"
    person_detector_model: str = "yolox_m_humanart"
    person_detector_min_score: float = 0.1
    pose_estimator_model: str = "rtmpose_s"
    analysis_empty_scene_fps: float = 1.0
    analysis_person_visible_fps: float = 5.0
    analysis_min_fps: float = 0.5
    tracker_new_track_min_score: float = 0.3
    tracker_lost_track_timeout_s: float = 3.0
    analysis_person_hold_s: float = 3.0
    alert_min_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    alert_language: Literal["en", "pt-BR"] = "en"
    alert_timezone: str = "UTC"
    alert_aggregation_window_s: float = Field(default=300.0, ge=0.0)
    alert_late_after_s: float = Field(default=60.0, gt=0.0)
    event_log_file: str = "/events/events.jsonl"
    outbox_dir: str = "/events/outbox"
    telegram_enabled: bool = False
    telegram_bot_token: SecretStr = SecretStr("")
    telegram_chat_ids: str = ""
    webhook_url: SecretStr | None = None
    webhook_timeout_s: float = Field(default=10.0, gt=0.0)
    health_check_interval_s: float = Field(default=5.0, gt=0.0)
    health_camera_lost_s: float = Field(default=30.0, gt=0.0)
    health_restore_after_s: float = Field(default=30.0, ge=0.0)
    health_camera_max_reconnects: int = Field(default=3, ge=1)
    health_reconnect_window_s: float = Field(default=600.0, gt=0.0)
    health_max_processing_failures: int = Field(default=5, ge=1)
    health_database_backlog_max_age_s: float = Field(default=300.0, gt=0.0)
    health_stall_exit_s: float = Field(default=180.0, ge=60.0)
    health_state_file: str = "/events/health_state.json"
    fall_detector_enabled: bool = True
    person_down_detector_enabled: bool = True
    person_down_detector_after_fall_s: float = Field(default=30.0, gt=0.0)
    person_down_detector_without_fall_s: float = Field(default=60.0, gt=0.0)
    head_impact_detector_enabled: bool = True
    head_impact_min_evidence: int = Field(default=2, ge=1)
    head_impact_window_s: float = Field(default=3.0, gt=0.0)
    audio_enabled: bool = True
    audio_thump_min_db: float = Field(default=12.0, gt=0.0)
    recordings_dir: str = "/recordings"
    event_clip_dir: str = "/events"
    recordings_max_age_h: float = Field(default=168.0, gt=0.0)
    recordings_max_gb: float = Field(default=200.0, gt=0.0)
    recordings_disk_min_free_gb: float = Field(default=10.0, ge=0.0)
    event_clip_pre_s: float = Field(default=20.0, ge=0.0)
    event_clip_post_s: float = Field(default=60.0, ge=0.0)
    event_clip_max_mb: float = Field(default=50.0, gt=0.0)
    event_clip_retention_days: float = Field(default=30.0, gt=0.0)
    db_host: str = "postgres"
    db_port: int = Field(default=5432, gt=0, lt=65536)
    db_name: str = "fim"
    db_user: str = "fim"
    db_password: SecretStr = SecretStr("")

    @field_validator("alert_timezone")
    @classmethod
    def _known_timezone(cls, value: str) -> str:
        try:
            ZoneInfo(value)
        except (ZoneInfoNotFoundError, ValueError) as error:
            raise ValueError(f"unknown time zone {value!r} (use an IANA name)") from error
        return value

    @model_validator(mode="after")
    def _telegram_complete(self) -> Self:
        if self.telegram_enabled:
            if not self.telegram_bot_token.get_secret_value():
                raise ValueError(
                    "FIM_TELEGRAM_BOT_TOKEN is required when FIM_TELEGRAM_ENABLED is true"
                )
            if not self.telegram_chat_id_list():
                raise ValueError(
                    "FIM_TELEGRAM_CHAT_IDS is required when FIM_TELEGRAM_ENABLED is true"
                )
        return self

    def telegram_chat_id_list(self) -> list[int]:
        """Chats the bot sends to and answers; every other chat is ignored."""
        try:
            return [int(part) for part in self.telegram_chat_ids.split(",") if part.strip()]
        except ValueError as error:
            raise ValueError("FIM_TELEGRAM_CHAT_IDS must be comma-separated integers") from error

    def database_url(self) -> str:
        """SQLAlchemy URL of the database (asyncpg). Only called by code that uses the
        database, so the alert path never depends on it being configured."""
        password = self.db_password.get_secret_value()
        if not password:
            raise ValueError("FIM_DB_PASSWORD is required to use the database")
        user, secret, name = (quote(v, safe="") for v in (self.db_user, password, self.db_name))
        return f"postgresql+asyncpg://{user}:{secret}@{self.db_host}:{self.db_port}/{name}"


@dataclass(frozen=True)
class CameraSettings:
    """Streams of one camera declared through environment variables."""

    id: str
    main_url: str
    sub_url: str | None
    zones_file: str | None = None


def load_cameras(env: Mapping[str, str]) -> list[CameraSettings]:
    """Read cameras declared in ``FIM_CAMERAS`` and their ``FIM_CAMERA_<ID>_*`` URLs.

    Error messages name variables, never their values (URLs may hold credentials).
    """
    declared = env.get("FIM_CAMERAS", "").strip()
    if not declared:
        return []
    cameras: list[CameraSettings] = []
    for raw_id in declared.split(","):
        camera_id = raw_id.strip()
        if not _CAMERA_ID.fullmatch(camera_id):
            raise ValueError(
                f"invalid camera id {camera_id!r} in FIM_CAMERAS: use lowercase letters, "
                "digits and underscores"
            )
        prefix = f"FIM_CAMERA_{camera_id.upper()}_"
        main_url = env.get(prefix + "MAIN_URL", "").strip()
        if not main_url:
            raise ValueError(f"{prefix}MAIN_URL is required for declared camera id {camera_id!r}")
        sub_url = env.get(prefix + "SUB_URL", "").strip() or None
        zones_file = env.get(prefix + "ZONES_FILE", "").strip() or None
        cameras.append(
            CameraSettings(id=camera_id, main_url=main_url, sub_url=sub_url, zones_file=zones_file)
        )
    ids = [camera.id for camera in cameras]
    for camera in cameras:
        if ids.count(camera.id) > 1:
            raise ValueError(f"camera id {camera.id!r} is declared twice in FIM_CAMERAS")
        if camera.sub_url and f"{camera.id}sub" in ids:
            raise ValueError(
                f"camera id {camera.id + 'sub'!r} clashes with the substream path of camera "
                f"{camera.id!r}; rename it"
            )
    return cameras
