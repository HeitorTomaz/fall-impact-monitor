"""Zones drawn per camera in normalized image coordinates (0..1, origin top-left).

``safe`` zones mark soft places (bed, sofa, a mattress on the floor, play mat). People
lying in them raise no floor event, a possible fall that ends in them is a near-miss,
and a fast head stop in them is not head-impact evidence. Everything else (walls,
furniture, the floor) is treated as a place where a head can hit something, so the
head-impact rules need no zone at all. ``suggested_safe`` zones are written by
``fim zones`` from detected furniture and are refused until the installer has checked each
one and changed its kind to ``safe``: a wrong safe zone silently turns alerts into
near-misses. Zone files describe a household and stay in git-ignored paths.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from fall_impact_monitor.perception.tracking import Box

ZoneKind = Literal["safe", "suggested_safe"]
ZONE_KINDS: tuple[ZoneKind, ...] = ("safe",)
SUGGESTED_KIND: ZoneKind = "suggested_safe"
_REPLACED_KINDS = {
    "allowed_lying": "rename the kind to 'safe'",
    "hazard": "remove the zone: everything outside safe zones is now treated as a hazard",
}


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


# Suggested zones are shrunk by this share of their box on each side: a box drawn around a
# bed or a sofa also covers some floor next to it, and a fall onto that floor must alert.
SUGGESTION_SHRINK = 0.1


def suggest_safe_zones(
    found: list[tuple[str, Box]], frame_size: tuple[int, int], shrink: float = SUGGESTION_SHRINK
) -> list[Zone]:
    """Suggested safe zones from detected soft furniture (label, box in pixels), shrunk
    towards their centre and named ``<label>_<n>``. The installer checks each one."""
    w, h = frame_size
    counts: dict[str, int] = {}
    zones = []
    for label, (x1, y1, x2, y2) in found:
        counts[label] = counts.get(label, 0) + 1
        dx, dy = shrink * (x2 - x1), shrink * (y2 - y1)
        left, top = round((x1 + dx) / w, 3), round((y1 + dy) / h, 3)
        right, bottom = round((x2 - dx) / w, 3), round((y2 - dy) / h, 3)
        polygon = ((left, top), (right, top), (right, bottom), (left, bottom))
        zones.append(Zone(f"{label}_{counts[label]}", SUGGESTED_KIND, polygon))
    return zones


def zone_file_json(zones: list[Zone], frame_size: tuple[int, int]) -> dict[str, object]:
    """The zone file format read by ``load_zones``."""
    return {
        "frame_size": list(frame_size),
        "zones": [
            {"name": z.name, "kind": z.kind, "polygon": [list(p) for p in z.polygon]} for z in zones
        ],
    }


def load_zones(path: Path, *, allow_suggested: bool = False) -> list[Zone]:
    """Read ``{"zones": [{"name", "kind", "polygon": [[x, y], ...]}]}``. Suggested zones
    are refused unless ``allow_suggested`` (``fim zones --check`` draws them)."""
    data = json.loads(path.read_text(encoding="utf-8"))
    zones = []
    for entry in data.get("zones", []):
        name = str(entry.get("name", ""))
        kind = entry.get("kind")
        if kind in _REPLACED_KINDS:
            hint = _REPLACED_KINDS[str(kind)]
            raise ValueError(f"zone {name!r}: kind {kind!r} was replaced by 'safe'; {hint}")
        if kind == SUGGESTED_KIND and not allow_suggested:
            raise ValueError(
                f"zone {name!r} is a suggestion: check it on zones.png, trim it to the soft "
                "surface, then change its kind to 'safe'"
            )
        if kind not in (*ZONE_KINDS, SUGGESTED_KIND):
            raise ValueError(f"zone {name!r}: kind must be one of {ZONE_KINDS}, got {kind!r}")
        polygon = tuple((float(x), float(y)) for x, y in entry.get("polygon", []))
        if len(polygon) < 3:
            raise ValueError(f"zone {name!r}: polygon needs at least 3 points")
        zones.append(Zone(name, cast(ZoneKind, kind), polygon))
    return zones


def zone_at(zones: list[Zone], box: Box, frame_size: tuple[int, int]) -> Zone | None:
    """The first zone that contains the centre of ``box`` (pixels), if any."""
    cx = (box[0] + box[2]) / 2 / frame_size[0]
    cy = (box[1] + box[3]) / 2 / frame_size[1]
    return next((zone for zone in zones if zone.contains(cx, cy)), None)
