"""Following people between person-detector runs with the pose estimator (#68).

Between two detector runs, each person's pose is estimated on their previous box
widened by ``SEARCH_MARGIN``. The new box is the box of the confident keypoints, widened
by the margins the detector's box had around that person's keypoints at the last
detector run, so that box sizes (which the rules normalise by) stay comparable.
"""

from dataclasses import dataclass

import numpy as np

from fall_impact_monitor.perception.pose import Box, Pose

KEYPOINT_MIN_SCORE = 0.3  # a keypoint counts towards the box from this score
MIN_KEYPOINTS = 5  # fewer confident keypoints: the pose is lost
MIN_KEYPOINT_BOX_PX = 1.0  # a keypoint box thinner than this cannot carry margins
SEARCH_MARGIN = 0.25  # fraction of the box's width and height added on each side
MIN_FOLLOW_IOU = 0.5  # a box that changed more than this between analysed frames is redetected
EDGE_PX = 2.0  # a box this close to the image border may be leaving: redetect

Margins = tuple[float, float, float, float]  # left, top, right, bottom; keypoint-box fractions


@dataclass(frozen=True)
class Followed:
    """A person carried to the next analysed frame without the detector."""

    box_xyxy: Box
    score: float  # the detector's score at its last run
    margins: Margins


def keypoint_box(pose: Pose) -> Box | None:
    """Box of the confident keypoints; ``None`` when too few are confident."""
    confident = pose.keypoints[pose.scores >= KEYPOINT_MIN_SCORE]
    if len(confident) < MIN_KEYPOINTS:
        return None
    x1, y1 = (float(v) for v in np.min(confident, axis=0))
    x2, y2 = (float(v) for v in np.max(confident, axis=0))
    if x2 - x1 < MIN_KEYPOINT_BOX_PX or y2 - y1 < MIN_KEYPOINT_BOX_PX:
        return None
    return (x1, y1, x2, y2)


def box_margins(detector_box: Box, kp_box: Box) -> Margins:
    width, height = kp_box[2] - kp_box[0], kp_box[3] - kp_box[1]
    return (
        (kp_box[0] - detector_box[0]) / width,
        (kp_box[1] - detector_box[1]) / height,
        (detector_box[2] - kp_box[2]) / width,
        (detector_box[3] - kp_box[3]) / height,
    )


def apply_margins(kp_box: Box, margins: Margins) -> Box:
    width, height = kp_box[2] - kp_box[0], kp_box[3] - kp_box[1]
    left, top, right, bottom = margins
    return (
        kp_box[0] - left * width,
        kp_box[1] - top * height,
        kp_box[2] + right * width,
        kp_box[3] + bottom * height,
    )


def search_box(box: Box, *, width: int, height: int) -> Box:
    """``box`` widened by ``SEARCH_MARGIN`` on each side, kept inside the image."""
    dx, dy = SEARCH_MARGIN * (box[2] - box[0]), SEARCH_MARGIN * (box[3] - box[1])
    return (
        max(0.0, box[0] - dx),
        max(0.0, box[1] - dy),
        min(width - 1.0, box[2] + dx),
        min(height - 1.0, box[3] + dy),
    )


def touches_edge(box: Box, *, width: int, height: int) -> bool:
    return (
        box[0] <= EDGE_PX
        or box[1] <= EDGE_PX
        or box[2] >= width - 1 - EDGE_PX
        or box[3] >= height - 1 - EDGE_PX
    )
