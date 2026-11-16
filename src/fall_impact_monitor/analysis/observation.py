"""Per-person geometric features for the fall rules.

Torso angle is measured between the shoulder midpoint and the hip midpoint
(0 = upright, 90 = lying, 180 = shoulders straight below the hips). Box aspect is
width / height. Hip and shoulder heights are image pixels; rules normalise them by the
person's height.
"""

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP = 5, 6, 11, 12

Box = tuple[float, float, float, float]


@dataclass(frozen=True)
class Observation:
    valid: bool  # hips and shoulders visible
    t_s: float
    aspect: float
    height_px: float
    person_score: float
    hip_y: float = 0.0
    shoulder_y: float = 0.0
    torso_angle_deg: float = 0.0


def _midpoint(
    points: NDArray[np.float32], scores: NDArray[np.float32], a: int, b: int, threshold: float
) -> tuple[float, float] | None:
    """Average of joints ``a`` and ``b``; one visible side alone; ``None`` if neither."""
    visible = [j for j in (a, b) if scores[j] >= threshold]
    if not visible:
        return None
    return (
        float(np.mean([points[j][0] for j in visible])),
        float(np.mean([points[j][1] for j in visible])),
    )


def observe(
    box: Box,
    keypoints: NDArray[np.float32] | None,
    keypoint_scores: NDArray[np.float32] | None,
    person_score: float,
    t_s: float,
    keypoint_threshold: float = 0.25,
) -> Observation:
    x1, y1, x2, y2 = box
    height = max(y2 - y1, 1e-6)
    aspect = (x2 - x1) / height
    base = Observation(False, t_s, aspect, height, person_score)
    if keypoints is None or keypoint_scores is None:
        return base
    hip = _midpoint(keypoints, keypoint_scores, LEFT_HIP, RIGHT_HIP, keypoint_threshold)
    shoulder = _midpoint(
        keypoints, keypoint_scores, LEFT_SHOULDER, RIGHT_SHOULDER, keypoint_threshold
    )
    if hip is None or shoulder is None:
        return base
    dx, dy = hip[0] - shoulder[0], hip[1] - shoulder[1]
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return base
    # Signed: shoulders below the hips (lying head first towards a high camera) read
    # as up to 180, never as upright.
    angle = math.degrees(math.atan2(abs(dx), dy))
    return Observation(True, t_s, aspect, height, person_score, hip[1], shoulder[1], angle)
