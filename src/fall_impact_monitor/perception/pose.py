"""Top-down pose estimation with RTMPose ONNX exports (COCO-17 keypoints)."""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from fall_impact_monitor._vendor.rtmlib import bbox_xyxy2cs, get_simcc_maximum, top_down_affine
from fall_impact_monitor.perception.detector import OnnxSession, create_session

INPUT_SIZE = (192, 256)  # model input (width, height)
BOX_PADDING = 1.25
SIMCC_SPLIT_RATIO = 2.0
MEAN = np.array([123.675, 116.28, 103.53], np.float32)  # RGB order
STD = np.array([58.395, 57.12, 57.375], np.float32)
MIN_BOX_SIDE_PX = 2.0

Box = tuple[float, float, float, float]


@dataclass(frozen=True)
class Pose:
    keypoints: NDArray[np.float32]  # (17, 2) image pixels, COCO-17 order
    scores: NDArray[np.float32]  # (17,)


def crop_geometry(box_xyxy: Box) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Center and scale of the model crop: padded box with the 3:4 aspect ratio enforced."""
    center, scale = bbox_xyxy2cs(np.array(box_xyxy, np.float64), padding=BOX_PADDING)
    width, height = INPUT_SIZE
    aspect = width / height
    if scale[0] > scale[1] * aspect:
        scale = np.array([scale[0], scale[0] / aspect])
    else:
        scale = np.array([scale[1] * aspect, scale[1]])
    return center, scale


def decode_keypoints(
    simcc_x: NDArray[np.floating],
    simcc_y: NDArray[np.floating],
    center: NDArray[np.floating],
    scale: NDArray[np.floating],
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Keypoints in image pixels and their scores, for one person's (K, W) SimCC maps."""
    locs, vals = get_simcc_maximum(simcc_x[None], simcc_y[None])
    keypoints = locs[0] / SIMCC_SPLIT_RATIO
    keypoints = keypoints / np.array(INPUT_SIZE) * scale + center - scale / 2
    return keypoints.astype(np.float32), vals[0].astype(np.float32)


class PoseEstimator:
    """Runs RTMPose on one crop per box, batched in a single inference call."""

    def __init__(self, session: OnnxSession) -> None:
        self._session = session
        self._input_name: str = session.get_inputs()[0].name

    @classmethod
    def from_path(cls, path: Path, providers: list[str] | None = None) -> "PoseEstimator":
        return cls(create_session(path, providers))

    def estimate(self, image_bgr: NDArray[np.uint8], boxes: Sequence[Box]) -> list[Pose | None]:
        """One pose per box; ``None`` for boxes too small to crop."""
        results: list[Pose | None] = [None] * len(boxes)
        valid = [
            i
            for i, (x1, y1, x2, y2) in enumerate(boxes)
            if x2 - x1 >= MIN_BOX_SIDE_PX and y2 - y1 >= MIN_BOX_SIDE_PX
        ]
        if not valid:
            return results
        crops = []
        geometry = []
        for i in valid:
            center, scale = crop_geometry(boxes[i])
            crop, _ = top_down_affine(INPUT_SIZE, scale, center, image_bgr)
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32)
            crops.append(((rgb - MEAN) / STD).transpose(2, 0, 1))
            geometry.append((center, scale))
        outputs = self._session.run(None, {self._input_name: np.stack(crops)})
        simcc_x, simcc_y = outputs[0], outputs[1]
        for row, i in enumerate(valid):
            center, scale = geometry[row]
            keypoints, scores = decode_keypoints(simcc_x[row], simcc_y[row], center, scale)
            results[i] = Pose(keypoints=keypoints, scores=scores)
        return results
