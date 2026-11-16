"""Pose windows: the input of temporal classifiers (#15).

A window is a person's last ``WINDOW_OBSERVATIONS`` analysed poses, oldest first. Only
observed frames are used, at their irregular times: each row carries the real time since
the previous row (``dt_s``) and ``observed`` = 1. A track with fewer observations is
padded with rows of zeros before them (``observed`` = 0); nothing is interpolated.

All rows share one reference so motion stays visible: coordinates are relative to the
hip midpoint of the window's first observation (its box centre when no hip is
confident) and divided by the person's tallest box height in the window. Image y grows
downwards, so a drop is an increasing y.

Row layout: for each COCO-17 keypoint (x, y, confidence), then ``dt_s``, ``observed``.
"""

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from fall_impact_monitor.api.perception import FrameResult

WINDOW_OBSERVATIONS = 16  # about 2 s at 8 analysed frames per second
KEYPOINTS = 17
FEATURES = KEYPOINTS * 3 + 2
HIP_MIN_SCORE = 0.3  # a hip counts towards the reference from this score
TRACK_MAX_GAP_S = 2.0  # a track unseen longer than this starts a new window
_LEFT_HIP, _RIGHT_HIP = 11, 12

Box = tuple[float, float, float, float]


@dataclass(frozen=True)
class PoseObservation:
    t_s: float
    box: Box
    keypoints: NDArray[np.float32]  # (17, 2) image pixels
    scores: NDArray[np.float32]  # (17,)


def _reference(o: PoseObservation) -> NDArray[np.float64]:
    hips = [j for j in (_LEFT_HIP, _RIGHT_HIP) if o.scores[j] >= HIP_MIN_SCORE]
    if hips:
        return np.mean(o.keypoints[hips].astype(np.float64), axis=0)
    x1, y1, x2, y2 = o.box
    return np.array([(x1 + x2) / 2, (y1 + y2) / 2])


def window_features(
    observations: Sequence[PoseObservation], length: int = WINDOW_OBSERVATIONS
) -> NDArray[np.float32]:
    """The ``(length, FEATURES)`` window of the latest ``observations`` (oldest first)."""
    kept = list(observations)[-length:]
    features = np.zeros((length, FEATURES), np.float32)
    if not kept:
        return features
    centre = _reference(kept[0])
    scale = max(max(o.box[3] - o.box[1] for o in kept), 1.0)
    first_row = length - len(kept)
    previous_s: float | None = None
    for row, o in enumerate(kept, start=first_row):
        points = (o.keypoints.astype(np.float64) - centre) / scale
        features[row, : KEYPOINTS * 3] = np.column_stack([points, o.scores]).reshape(-1)
        features[row, -2] = 0.0 if previous_s is None else o.t_s - previous_s
        features[row, -1] = 1.0
        previous_s = o.t_s
    return features


class PoseWindows:
    """Keeps each track's latest observations; one window per track with keypoints in
    every analysed frame."""

    def __init__(
        self, length: int = WINDOW_OBSERVATIONS, max_gap_s: float = TRACK_MAX_GAP_S
    ) -> None:
        self._length = length
        self._max_gap_s = max_gap_s
        self._tracks: dict[int, deque[PoseObservation]] = {}

    def update(self, result: FrameResult) -> list[tuple[int, NDArray[np.float32]]]:
        now_s = result.record.arrival_ns / 1e9
        windows = []
        for track in result.tracks:
            if track.keypoints is None or track.keypoint_scores is None:
                continue
            history = self._tracks.get(track.track_id)
            if history is None or now_s - history[-1].t_s > self._max_gap_s:
                history = deque(maxlen=self._length)
                self._tracks[track.track_id] = history
            history.append(
                PoseObservation(now_s, track.box_xyxy, track.keypoints, track.keypoint_scores)
            )
            windows.append((track.track_id, window_features(history, self._length)))
        stale = [i for i, h in self._tracks.items() if now_s - h[-1].t_s > self._max_gap_s]
        for track_id in stale:
            del self._tracks[track_id]
        return windows
