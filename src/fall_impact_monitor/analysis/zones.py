"""Zones drawn per camera in normalized image coordinates (0..1, origin top-left).

``allowed_lying`` zones (sofa, bed, play mat) exempt people lying in them from the
floor rule; ``hazard`` zones (walls, furniture) are used by the head-impact rules.
Zone files describe a household and stay in git-ignored paths.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

ZoneKind = Literal["allowed_lying", "hazard"]
ZONE_KINDS: tuple[ZoneKind, ...] = ("allowed_lying", "hazard")


@dataclass(frozen=True)
class Zone:
    name: str
    kind: ZoneKind
    polygon: tuple[tuple[float, float], ...]

    def contains(self, x: float, y: float) -> bool:
        """Ray-casting point-in-polygon test in normalized coordinates."""
        inside = False
        points = self.polygon
        for i, (x1, y1) in enumerate(points):
            x2, y2 = points[(i + 1) % len(points)]
            if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
                inside = not inside
        return inside


def load_zones(path: Path) -> list[Zone]:
    """Read ``{"zones": [{"name", "kind", "polygon": [[x, y], ...]}]}``."""
    data = json.loads(path.read_text(encoding="utf-8"))
    zones = []
    for entry in data.get("zones", []):
        name = str(entry.get("name", ""))
        kind = entry.get("kind")
        if kind not in ZONE_KINDS:
            raise ValueError(f"zone {name!r}: kind must be one of {ZONE_KINDS}, got {kind!r}")
        polygon = tuple((float(x), float(y)) for x, y in entry.get("polygon", []))
        if len(polygon) < 3:
            raise ValueError(f"zone {name!r}: polygon needs at least 3 points")
        zones.append(Zone(name, cast(ZoneKind, kind), polygon))
    return zones
