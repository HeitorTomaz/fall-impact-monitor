"""Application settings read from ``FIM_*`` environment variables."""

import re
from collections.abc import Mapping
from dataclasses import dataclass

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
    alert_confidence_threshold: float = 0.5
    fall_detector_enabled: bool = True


@dataclass(frozen=True)
class CameraSettings:
    """Streams of one camera declared through environment variables."""

    id: str
    main_url: str
    sub_url: str | None


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
        cameras.append(CameraSettings(id=camera_id, main_url=main_url, sub_url=sub_url))
    return cameras
