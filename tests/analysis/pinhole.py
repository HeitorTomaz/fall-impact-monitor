"""Synthetic people seen by a pinhole camera mounted at a given height (#78).

A 1.7 m person stands ``distance_m`` in front of the camera; the camera is aimed at
their hips. Poses are 3D COCO-17 joints in metres (x right, y away from the camera, z
up), projected to a 1280 x 720 image. Falls rotate the body about the ankles; daily
activities move the joints to a sitting or crouching pose.
"""

import math
from datetime import UTC, datetime
from fractions import Fraction

import numpy as np
from numpy.typing import NDArray

from fall_impact_monitor.api.perception import FrameResult
from fall_impact_monitor.capture import FrameRecord, TimestampQuality
from fall_impact_monitor.perception.tracking import Track

FPS = 8.0
SIZE = (1280, 720)
_FOCAL_PX = 600.0

# Standing COCO-17 joints: (x, z) in metres; y is the person's distance.
_STANDING = np.array(
    [
        (0.0, 1.62), (-0.03, 1.65), (0.03, 1.65), (-0.07, 1.63), (0.07, 1.63),  # head
        (-0.2, 1.4), (0.2, 1.4),  # shoulders
        (-0.25, 1.1), (0.25, 1.1),  # elbows
        (-0.27, 0.85), (0.27, 0.85),  # wrists
        (-0.12, 0.9), (0.12, 0.9),  # hips
        (-0.12, 0.5), (0.12, 0.5),  # knees
        (-0.12, 0.05), (0.12, 0.05),  # ankles
    ]
)  # fmt: skip


def standing(distance_m: float) -> NDArray[np.float64]:
    """(17, 3) joints of a person standing ``distance_m`` from the camera."""
    out = np.zeros((17, 3))
    out[:, 0], out[:, 1], out[:, 2] = _STANDING[:, 0], distance_m, _STANDING[:, 1]
    return out


def fallen(distance_m: float, direction: str, k: float) -> NDArray[np.float64]:
    """The body rotated ``k`` of 90 degrees about the ankles: ``away`` from the camera,
    ``toward`` it or ``sideways``; lying bodies stay 0.1 m above the floor."""
    joints = standing(distance_m)
    phi = math.radians(90.0 * k)
    x, z = joints[:, 0].copy(), joints[:, 2].copy()
    if direction == "sideways":  # the left-right axis turns vertical: the body keeps a height
        joints[:, 0] = x * math.cos(phi) + z * math.sin(phi)
        joints[:, 2] = -x * math.sin(phi) + z * math.cos(phi) + 0.3 * math.sin(phi)
        return joints
    along = z * math.sin(phi)
    joints[:, 1] += along if direction == "away" else -along
    joints[:, 2] = np.maximum(z * math.cos(phi), 0.1 * k)
    return joints


def sitting(distance_m: float, k: float) -> NDArray[np.float64]:
    """Sitting down on a chair behind the person (``k`` from 0 to 1): hips go 0.45 m
    down and 0.3 m back, the trunk stays upright, the knees come forward."""
    joints = standing(distance_m)
    upper = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    joints[upper, 2] -= 0.45 * k
    joints[upper, 1] += 0.3 * k
    joints[[13, 14], 1] -= 0.2 * k
    return joints


def crouching(distance_m: float, k: float) -> NDArray[np.float64]:
    """Crouching to pick something up: hips 0.5 m down, trunk leaning forward 30 degrees."""
    joints = standing(distance_m)
    lean = math.radians(30.0 * k)
    hip_z = 0.9 - 0.5 * k
    for j in range(13):
        above = joints[j, 2] - 0.9
        joints[j, 2] = hip_z + above * math.cos(lean)
        joints[j, 1] -= above * math.sin(lean)
    joints[[13, 14], 2] = 0.5 - 0.15 * k
    joints[[13, 14], 1] -= 0.25 * k
    return joints


def project(
    joints: NDArray[np.float64], camera_height_m: float, distance_m: float
) -> NDArray[np.float32]:
    """Image points of the joints; the camera is aimed at the standing person's hips."""
    forward = np.array([0.0, distance_m, 0.9 - camera_height_m])
    forward /= np.linalg.norm(forward)
    right = np.array([1.0, 0.0, 0.0])
    up = np.cross(right, forward)
    rel = joints - np.array([0.0, 0.0, camera_height_m])
    depth = rel @ forward
    x = SIZE[0] / 2 + _FOCAL_PX * (rel @ right) / depth
    y = SIZE[1] / 2 - _FOCAL_PX * (rel @ up) / depth
    return np.stack([x, y], axis=1).astype(np.float32)


def track(points: NDArray[np.float32], track_id: int = 1) -> Track:
    x1, y1 = points.min(axis=0)
    x2, y2 = points.max(axis=0)
    pad = 0.08 * (y2 - y1)
    box = (float(x1 - pad), float(y1 - pad), float(x2 + pad), float(y2 + pad))
    scores = np.full(17, 0.9, np.float32)
    return Track(track_id, box, 0.9, points, scores, "stable", 30, 0)


def frame(seq: int, tracks: list[Track]) -> FrameResult:
    t_s = seq / FPS
    record = FrameRecord(
        "cam1",
        seq,
        seq * 11250,
        Fraction(1, 90000),
        int(t_s * 1e9),
        datetime(2026, 1, 1, tzinfo=UTC),
        False,
        100,
        TimestampQuality.OK,
    )
    return FrameResult(record, tracks, analysed=True)


def scenario(
    pose_at: "callable[[float], NDArray[np.float64]]",  # type: ignore[valid-type]
    camera_height_m: float,
    distance_m: float,
    *,
    stand_s: float = 2.0,
    change_s: float = 0.8,
    hold_s: float = 3.0,
) -> list[FrameResult]:
    """Frames at ``FPS``: standing, then ``pose_at(k)`` for k from 0 to 1 over
    ``change_s``, then held at k = 1."""
    frames = []
    total = stand_s + change_s + hold_s
    for seq in range(int(total * FPS)):
        t_s = seq / FPS
        k = min(1.0, max(0.0, (t_s - stand_s) / change_s))
        frames.append(frame(seq, [track(project(pose_at(k), camera_height_m, distance_m))]))
    return frames
