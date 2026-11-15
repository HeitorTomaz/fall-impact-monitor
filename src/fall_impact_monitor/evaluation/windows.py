"""Labelled pose windows from saved ``fim eval`` analyses, for temporal classifiers (#15).

Every analysed frame gives one window per track with keypoints (``analysis.pose_window``),
labelled against the dataset's falls: 1 when its time span overlaps a fall, 0 when it
does not, and left out when it ends within ``IGNORE_AFTER_S`` after a fall (the person
lying right after a fall is neither). Frames ``fim eval`` adds after a clip's end (its last
frame repeated, flagged ``padded``) give no window. Each window carries a group (the person, or the
event for datasets without subject labels) so that splits never put one person or one
event on both sides.
"""

import json
import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from fall_impact_monitor.analysis.pose_window import PoseWindows
from fall_impact_monitor.evaluation.truth import TruthEvent

IGNORE_AFTER_S = 1.0

_GROUP = {
    "gmdcsa24": re.compile(r"^(Subject \d+)/"),
    "caucafall": re.compile(r"(Subject\.\d+)/"),
    "urfd": re.compile(r"((?:fall|adl)-\d+)-cam"),
    "mcfd": re.compile(r"(chute\d+)/"),
}


def window_label(start_s: float, end_s: float, falls: Sequence[TruthEvent]) -> int | None:
    if any(start_s <= f.end_s and end_s >= f.start_s for f in falls):
        return 1
    if any(f.end_s < end_s <= f.end_s + IGNORE_AFTER_S for f in falls):
        return None
    return 0


def group_of(dataset: str, video_name: str) -> str:
    match = _GROUP[dataset].search(video_name)
    if match is None:
        raise ValueError(f"{video_name}: no group for {dataset}")
    return f"{dataset}:{match[1]}"


@dataclass(frozen=True)
class Window:
    features: NDArray[np.float32]
    label: int
    track_id: int
    end_s: float  # from the video's first frame


def labelled_windows(
    entries: Iterator[dict[str, object]], falls: Sequence[TruthEvent], start_s: float
) -> list[Window]:
    """Windows of one video's saved ``tracks.jsonl`` lines; ``start_s``: media time of
    its first frame (truth times count from it)."""
    from fall_impact_monitor.cli.analyze import result_from_entry

    windows = PoseWindows()
    labelled = []
    for entry in entries:
        if not entry["analysed"] or entry.get("padded"):
            continue  # padded frames: the clip's last frame repeated by fim eval (#81)
        result, _ = result_from_entry(entry)
        end_s = result.record.arrival_ns / 1e9 - start_s
        for track_id, features in windows.update(result):
            span_s = float(features[:, -2].sum())  # the first observed row's gap is 0
            label = window_label(end_s - span_s, end_s, falls)
            if label is not None:
                labelled.append(Window(features, label, track_id, end_s))
    return labelled


def save_windows(
    path: Path,
    windows: Sequence[Window],
    groups: Sequence[str],
    videos: Sequence[str],
    meta: dict[str, object],
) -> None:
    np.savez_compressed(
        path,
        features=np.stack([w.features for w in windows]).astype(np.float32),
        labels=np.array([w.label for w in windows], np.int8),
        track_ids=np.array([w.track_id for w in windows], np.int32),
        end_s=np.array([w.end_s for w in windows], np.float64),
        groups=np.array(groups, dtype=str),
        videos=np.array(videos, dtype=str),
        meta=np.array(json.dumps(meta)),
    )
