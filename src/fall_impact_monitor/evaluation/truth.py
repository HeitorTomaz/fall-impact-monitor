"""Ground-truth falls of the public evaluation datasets, as time ranges per video.

Datasets are downloaded by the installer and read locally; nothing here is redistributed
(see docs/DATASETS.md). Each loader returns every video of the dataset, falls and
activities of daily living alike; a video without falls has an empty ``falls`` tuple.

- GMDCSA24: ``Subject N/{Fall,ADL}.csv``, last column, e.g. ``Falling (SW)[3.4 to 6]``.
  A fall class without a range (one row) lasts from the end of the row's last range to
  the end of the clip (the ``Length`` column).
- UR Fall (cam0): per-frame labels, ``-1`` not lying, ``0`` falling, ``1`` lying. A fall
  starts at its first ``0`` frame and ends at its last ``0`` or ``1`` frame. ADL sequences
  also contain ``0``/``1`` (lying down on purpose): they are not falls.
- CAUCAFall: per-frame YOLO labels next to each video, class ``1`` = fall, from the fall's
  onset to the end of the clip.
"""

import csv
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from fall_impact_monitor.evaluation.caucafall import fall_frames


@dataclass(frozen=True)
class TruthEvent:
    start_s: float
    end_s: float


@dataclass(frozen=True)
class LabelledVideo:
    name: str  # path relative to the dataset root, with forward slashes
    path: Path
    falls: tuple[TruthEvent, ...]


# A range is "[a to b]"; one row writes "[a b]".
_RANGE = r"\[\s*([\d.]+)\s*(?:to|-)?\s*([\d.]+)\s*\]"
_GMDCSA_FALL = re.compile(r"\bFall\w*[^\[;]*" + _RANGE, re.IGNORECASE)
_GMDCSA_FALL_NO_RANGE = re.compile(r"\bFall\w*[^\[;]*(?:;|$)", re.IGNORECASE)
_GMDCSA_RANGE = re.compile(_RANGE)


def _name(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _merged(falls: list[TruthEvent]) -> tuple[TruthEvent, ...]:
    """Falls that touch or overlap are one fall ("[2 to 4]; [4 to 11]")."""
    merged: list[TruthEvent] = []
    for fall in sorted(falls, key=lambda f: f.start_s):
        if merged and fall.start_s <= merged[-1].end_s:
            merged[-1] = TruthEvent(merged[-1].start_s, max(merged[-1].end_s, fall.end_s))
        else:
            merged.append(fall)
    return tuple(merged)


def gmdcsa24_videos(root: Path) -> list[LabelledVideo]:
    videos = []
    for table in sorted(root.glob("Subject */*.csv")):
        folder = table.parent / table.stem  # Fall.csv -> Fall/, ADL.csv -> ADL/
        with table.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.reader(handle))
        for row in rows[1:]:
            if not row or not row[0].strip():
                continue
            path = folder / row[0].strip()
            classes = row[-1]
            falls = [TruthEvent(float(a), float(b)) for a, b in _GMDCSA_FALL.findall(classes)]
            if not falls and _GMDCSA_FALL_NO_RANGE.search(classes):
                ends = [float(end) for _, end in _GMDCSA_RANGE.findall(classes)]
                falls = [TruthEvent(max(ends, default=0.0), float(row[1]))]
            if table.stem == "Fall" and not falls:
                raise ValueError(f"{table.parent.name}/Fall.csv: no fall read for {row[0]}")
            if path.exists():
                videos.append(LabelledVideo(_name(path, root), path, _merged(falls)))
    return sorted(videos, key=lambda v: v.name)


def _urfd_labels(table: Path) -> dict[str, list[tuple[int, int]]]:
    """Sequence name -> [(frame number, label)]."""
    sequences: dict[str, list[tuple[int, int]]] = {}
    with table.open(encoding="utf-8", newline="") as handle:
        for row in csv.reader(handle):
            if len(row) >= 3:
                sequences.setdefault(row[0], []).append((int(row[1]), int(row[2])))
    return sequences


def urfd_videos(root: Path, *, fps: float = 30.0) -> list[LabelledVideo]:
    falls = _urfd_labels(root / "raw" / "labels" / "urfall-cam0-falls.csv")
    videos = []
    for path in sorted((root / "raw" / "videos").glob("*-cam0.mp4")):
        sequence = path.name.removesuffix("-cam0.mp4")
        events: tuple[TruthEvent, ...] = ()
        frames = falls.get(sequence, [])
        falling = [frame for frame, label in frames if label == 0]
        if sequence.startswith("fall-") and falling:
            last = max(frame for frame, label in frames if label in (0, 1))
            events = (TruthEvent((min(falling) - 1) / fps, last / fps),)
        videos.append(LabelledVideo(_name(path, root), path, events))
    return videos


def caucafall_videos(root: Path, *, fps_of: Callable[[Path], float]) -> list[LabelledVideo]:
    videos = []
    for path in sorted(root.rglob("*.avi")):
        frames = fall_frames(path)  # 0-based
        events: tuple[TruthEvent, ...] = ()
        if frames:
            fps = fps_of(path)
            events = (TruthEvent(min(frames) / fps, (max(frames) + 1) / fps),)
        videos.append(LabelledVideo(_name(path, root), path, events))
    return videos
