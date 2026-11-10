"""`fim zones`: write a snapshot, suggested safe zones and an overview; validate a file.

Safe zones are suggested from beds and sofas detected in the snapshot by a COCO-trained
YOLOX model; the installer checks them on ``zones.png``, edits ``zones.json`` by hand
(mattresses on the floor and play mats are not detected) and runs ``--check`` to redraw
the overview. Coordinates are normalized (0..1, origin top-left), so a zone file survives
a change of stream resolution. Zone files describe a household: keep them in git-ignored
paths.
"""

import argparse
import json
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from fall_impact_monitor.analysis.zones import (
    Zone,
    load_zones,
    suggest_safe_zones,
    zone_file_json,
)
from fall_impact_monitor.capture.reader import SourceOptions, iter_frame_records
from fall_impact_monitor.config import mask_url
from fall_impact_monitor.perception.detector import (
    COCO_BED_CLASS,
    COCO_COUCH_CLASS,
    PersonDetector,
    create_session,
)
from fall_impact_monitor.perception.models import ModelError, load_manifest, resolve
from fall_impact_monitor.perception.tracking import Box

FurnitureDetector = Callable[[NDArray[np.uint8]], list[tuple[str, Box]]]
_FURNITURE = {"bed": COCO_BED_CLASS, "sofa": COCO_COUCH_CLASS}
_ZONE_COLOUR_BGR = (0, 200, 0)


def add_parser(subparsers: "argparse._SubParsersAction[argparse.ArgumentParser]") -> None:
    parser = subparsers.add_parser(
        "zones", help="write a snapshot, suggested safe zones and an overview"
    )
    parser.add_argument("--source", help="file or stream to take the snapshot from")
    parser.add_argument("--camera-id", default="zones")
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/zones"))
    parser.add_argument("--check", type=Path, help="validate a zone file, redraw zones.png")
    parser.add_argument("--manifest", type=Path, default=Path("models.toml"))
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--model", default="yolox_s", help="COCO-trained YOLOX model")
    parser.add_argument(
        "--min-score", type=float, default=0.3, help="lowest score of a suggested bed or sofa"
    )
    parser.set_defaults(handler=run)


def furniture_detector(args: argparse.Namespace) -> FurnitureDetector | None:
    """Beds and sofas in an image, or None (with a hint) when the model is missing."""
    try:
        path = resolve(load_manifest(args.manifest)[args.model], args.models_dir)
    except (ModelError, KeyError, OSError) as error:
        print(f"zones: no suggestions ({error}); run: fim models fetch --only {args.model}")
        return None
    session = create_session(path)
    detectors = {
        label: PersonDetector(session, score_threshold=args.min_score, class_id=class_id)
        for label, class_id in _FURNITURE.items()
    }

    def detect(image: NDArray[np.uint8]) -> list[tuple[str, Box]]:
        return [(label, d.box_xyxy) for label, det in detectors.items() for d in det.detect(image)]

    return detect


def draw_zones(image: NDArray[np.uint8], zones: list[Zone]) -> NDArray[np.uint8]:
    out = image.copy()
    h, w = out.shape[:2]
    for zone in zones:
        points = np.array([[round(x * w), round(y * h)] for x, y in zone.polygon], np.int32)
        cv2.polylines(out, [points], isClosed=True, color=_ZONE_COLOUR_BGR, thickness=2)
        x, y = points.min(axis=0)
        cv2.putText(
            out,
            zone.name,
            (int(x) + 4, int(y) + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            _ZONE_COLOUR_BGR,
            2,
        )
    return out


def _write_overview(zone_file: Path, zones: list[Zone]) -> None:
    snapshot = zone_file.parent / "snapshot.png"
    image = cv2.imread(str(snapshot)) if snapshot.exists() else None
    if image is not None:
        overview = draw_zones(np.asarray(image, dtype=np.uint8), zones)
        cv2.imwrite(str(zone_file.parent / "zones.png"), overview)


def _check(path: Path) -> int:
    try:
        zones = load_zones(path)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"zones: invalid {path.name}: {error}")
        return 1
    for zone in zones:
        print(f"zones: {zone.kind:6} {zone.name} ({len(zone.polygon)} points)")
    _write_overview(path, zones)
    return 0


def run(args: argparse.Namespace) -> int:
    if args.check is not None:
        return _check(args.check)
    if not args.source:
        print("zones: --source or --check is required")
        return 2
    image = None
    for _, packet in iter_frame_records(args.source, args.camera_id, SourceOptions()):
        frames = packet.decode()
        if frames:
            image = np.asarray(frames[0].to_ndarray(format="bgr24"), dtype=np.uint8)
            break
    if image is None:
        print(f"zones: no frame decoded from {mask_url(args.source)}")
        return 1
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out_dir / "snapshot.png"), image)
    zone_file = args.out_dir / "zones.json"
    if zone_file.exists():
        print(f"zones: kept existing {zone_file}")
        _check(zone_file)
        return 0
    size = (image.shape[1], image.shape[0])
    detector = furniture_detector(args)
    zones = suggest_safe_zones(detector(image), size) if detector is not None else []
    zone_file.write_text(json.dumps(zone_file_json(zones, size), indent=2) + "\n", "utf-8")
    _write_overview(zone_file, zones)
    print(f"zones: {len(zones)} suggested safe zone(s) in {zone_file}")
    print(
        "zones: check zones.png; keep each zone on the soft surface only, add mattresses "
        "on the floor by hand, then run fim zones --check"
    )
    return 0
