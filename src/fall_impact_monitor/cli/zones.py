"""`fim zones`: write a snapshot and a zone-file template to edit by hand; validate a file.

Coordinates are normalized (0..1, origin top-left), so a zone file survives a change
of stream resolution. Zone files describe a household: keep them in git-ignored paths.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from fall_impact_monitor.analysis.zones import load_zones
from fall_impact_monitor.capture.reader import SourceOptions, iter_frame_records
from fall_impact_monitor.config import mask_url

_TEMPLATE_ZONES = [
    {
        "name": "example_sofa",
        "kind": "allowed_lying",
        "polygon": [[0.05, 0.55], [0.35, 0.55], [0.35, 0.9], [0.05, 0.9]],
    },
    {
        "name": "example_wall",
        "kind": "hazard",
        "polygon": [[0.9, 0.0], [1.0, 0.0], [1.0, 1.0], [0.9, 1.0]],
    },
]


def add_parser(subparsers: "argparse._SubParsersAction[argparse.ArgumentParser]") -> None:
    parser = subparsers.add_parser("zones", help="write a snapshot and a zone-file template")
    parser.add_argument("--source", help="file or stream to take the snapshot from")
    parser.add_argument("--camera-id", default="zones")
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/zones"))
    parser.add_argument("--check", type=Path, help="validate a zone file and exit")
    parser.set_defaults(handler=run)


def _check(path: Path) -> int:
    try:
        zones = load_zones(path)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"zones: invalid {path.name}: {error}")
        return 1
    for zone in zones:
        print(f"zones: {zone.kind:14} {zone.name} ({len(zone.polygon)} points)")
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
    else:
        template = {"frame_size": [image.shape[1], image.shape[0]], "zones": _TEMPLATE_ZONES}
        zone_file.write_text(json.dumps(template, indent=2) + "\n", encoding="utf-8")
    print(f"zones: snapshot and template in {args.out_dir}")
    return 0
