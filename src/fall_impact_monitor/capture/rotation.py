"""Rotation of decoded frames for a camera mounted upside down (``FIM_CAMERA_<ID>_ROTATE_DEG``).

Applied before analysis only: recordings and event clips stay as the camera sends them.
"""

import numpy as np
from numpy.typing import NDArray

ROTATIONS_DEG = (0, 180)


def rotate(image: NDArray[np.uint8], rotate_deg: int) -> NDArray[np.uint8]:
    if rotate_deg == 0:
        return image
    if rotate_deg == 180:
        return np.ascontiguousarray(image[::-1, ::-1])
    raise ValueError(f"unsupported rotation {rotate_deg}; use one of {ROTATIONS_DEG}")
