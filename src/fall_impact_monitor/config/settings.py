"""Application settings read from ``FIM_*`` environment variables."""

import re
from collections.abc import Mapping
from dataclasses import dataclass
from urllib.parse import quote

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

_CAMERA_ID = re.compile(r"[a-z0-9_]+")


class Settings(BaseSettings):
    """Process-wide settings. Every field maps to ``FIM_<FIELD_NAME>``."""

    model_config = SettingsConfigDict(env_prefix="FIM_", env_file=".env", extra="ignore")

    log_level: str = "INFO"
    stale_timeout_s: float = 10.0
    packet_buffer_s: float = 20.0
    mediamtx_api_url: str = "http://mediamtx:9996"
    models_dir: str = "models"
    detector_model: str = "yolox_m_humanart"
    detector_score_threshold: float = 0.1
    pose_model: str = "rtmpose_s"
    empty_fps: float = 1.0
    person_fps: float = 5.0
    min_analysis_fps: float = 0.5
    new_track_min_score: float = 0.3
    track_lost_s: float = 3.0
    person_hold_s: float = 3.0
    alert_confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    fall_detector_enabled: bool = True
    floor_detector_enabled: bool = True
    person_down_s: float = Field(default=30.0, gt=0.0)
    floor_no_fall_s: float = Field(default=60.0, gt=0.0)
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
