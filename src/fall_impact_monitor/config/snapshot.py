"""Snapshot and hash of the active configuration, stored with every event.

Secrets never enter the snapshot: secret settings are masked, camera URLs lose their
credentials and the Telegram chat ids are reduced to a count.
"""

import hashlib
import json
from typing import Any

from fall_impact_monitor.config.masking import mask_url
from fall_impact_monitor.config.settings import CameraSettings, Settings


def config_snapshot(settings: Settings, cameras: list[CameraSettings]) -> dict[str, Any]:
    values = settings.model_dump(mode="json")
    values["telegram_chat_ids"] = f"{len(settings.telegram_chat_id_list())} chats"
    return {
        "settings": values,
        "cameras": [
            {
                "id": camera.id,
                "main_url": mask_url(camera.main_url),
                "sub_url": mask_url(camera.sub_url) if camera.sub_url else None,
                "zones_file": camera.zones_file,
                "rotate_deg": camera.rotate_deg,
            }
            for camera in cameras
        ],
    }


def config_hash(snapshot: dict[str, Any]) -> str:
    canonical = json.dumps(snapshot, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
