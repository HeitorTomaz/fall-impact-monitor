"""CAUCAFall labels (per-frame YOLO boxes, class 1 = fall) and box-level recall.

Each video folder holds one label file per frame; the last five digits of the
file name are the 1-based frame number. Labels are only read locally; the
dataset is never redistributed.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from fall_impact_monitor.perception.tracking import box_iou

Box = tuple[float, float, float, float]
_FRAME_FILE = re.compile(r"^[a-z]+(\d+)\.txt$")


@dataclass(frozen=True)
class LabeledBox:
    box_xyxy: Box
    fall: bool


def load_caucafall_labels(
    video_path: Path, *, width: int, height: int
) -> dict[int, list[LabeledBox]]:
    """Map 0-based frame index to the labelled people in that frame."""
    folder = video_path.parent
    if not folder.is_dir():
        raise FileNotFoundError(folder)
    labels: dict[int, list[LabeledBox]] = {}
    for path in folder.iterdir():
        match = _FRAME_FILE.match(path.name)
        if not match:
            continue
        frame = int(match.group(1)[-5:]) - 1
        boxes = []
        for line in path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            cls, cx, cy, w, h = int(parts[0]), *(float(p) for p in parts[1:])
            boxes.append(
                LabeledBox(
                    (
                        (cx - w / 2) * width,
                        (cy - h / 2) * height,
                        (cx + w / 2) * width,
                        (cy + h / 2) * height,
                    ),
                    fall=cls == 1,
                )
            )
        labels[frame] = boxes
    return labels


def match_recall(truth: Sequence[Box], detections: Sequence[Box], *, iou: float) -> tuple[int, int]:
    """(matched, total) ground-truth boxes; greedy one-to-one matching by IoU."""
    pairs = sorted(
        ((box_iou(t, d), ti, di) for ti, t in enumerate(truth) for di, d in enumerate(detections)),
        reverse=True,
    )
    used_truth: set[int] = set()
    used_detections: set[int] = set()
    for value, ti, di in pairs:
        if value < iou:
            break
        if ti in used_truth or di in used_detections:
            continue
        used_truth.add(ti)
        used_detections.add(di)
    return len(used_truth), len(truth)


def select_frames(labels: dict[int, list[LabeledBox]], count: int) -> list[int]:
    """Up to ``count`` labelled frames, half with a fall label and half without,
    evenly spaced within each group."""

    def spread(frames: list[int], n: int) -> list[int]:
        if n <= 0 or not frames:
            return []
        if len(frames) <= n:
            return frames
        step = len(frames) / n
        return [frames[int(i * step)] for i in range(n)]

    fall = sorted(f for f, boxes in labels.items() if any(b.fall for b in boxes))
    other = sorted(f for f, boxes in labels.items() if boxes and not any(b.fall for b in boxes))
    half = count // 2
    chosen = spread(fall, half) + spread(other, count - half)
    return sorted(chosen)
