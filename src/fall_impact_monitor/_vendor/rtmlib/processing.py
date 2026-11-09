# Copyright 2018-2020 Open-MMLab.
# Copyright 2026 The Fall Impact Monitor contributors
# SPDX-License-Identifier: Apache-2.0
#
# Adapted from https://github.com/Tau-J/rtmlib
#   @5ea82df885d6f2a66923f1de2d15c799af0ed732
#   rtmlib/tools/pose_estimation/pre_processings.py
#     (bbox_xyxy2cs, _rotate_point, _get_3rd_point, get_warp_matrix, top_down_affine)
#   rtmlib/tools/pose_estimation/post_processings.py (get_simcc_maximum)
# Changed: type annotations, float32 arrays passed directly to getAffineTransform, shorter
# docstrings, the "mean" score combination inlined,
# formatting; other helpers removed.

import cv2
import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.floating]


def bbox_xyxy2cs(bbox: FloatArray, padding: float = 1.0) -> tuple[FloatArray, FloatArray]:
    """Transform bboxes from (left, top, right, bottom) into (center, scale)."""
    dim = bbox.ndim
    if dim == 1:
        bbox = bbox[None, :]

    x1, y1, x2, y2 = np.hsplit(bbox, [1, 2, 3])
    center = np.hstack([x1 + x2, y1 + y2]) * 0.5
    scale = np.hstack([x2 - x1, y2 - y1]) * padding

    if dim == 1:
        center = center[0]
        scale = scale[0]

    return center, scale


def _rotate_point(pt: FloatArray, angle_rad: float) -> FloatArray:
    sn, cs = np.sin(angle_rad), np.cos(angle_rad)
    rot_mat = np.array([[cs, -sn], [sn, cs]])
    result: FloatArray = rot_mat @ pt
    return result


def _get_3rd_point(a: FloatArray, b: FloatArray) -> FloatArray:
    """Rotate vector ``a - b`` by 90 degrees anticlockwise around ``b``."""
    direction = a - b
    c: FloatArray = b + np.r_[-direction[1], direction[0]]
    return c


def get_warp_matrix(
    center: FloatArray,
    scale: FloatArray,
    rot: float,
    output_size: tuple[int, int],
    shift: tuple[float, float] = (0.0, 0.0),
    inv: bool = False,
) -> FloatArray:
    """Affine matrix warping the bbox area of the input image to ``output_size``."""
    shift_arr = np.array(shift)
    src_w = scale[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.deg2rad(rot)
    src_dir = _rotate_point(np.array([0.0, src_w * -0.5]), rot_rad)
    dst_dir = np.array([0.0, dst_w * -0.5])

    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale * shift_arr
    src[1, :] = center + src_dir + scale * shift_arr
    src[2, :] = _get_3rd_point(src[0, :], src[1, :])

    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir
    dst[2, :] = _get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        warp_mat = cv2.getAffineTransform(dst, src)
    else:
        warp_mat = cv2.getAffineTransform(src, dst)
    result: FloatArray = np.asarray(warp_mat, dtype=np.float64)
    return result


def top_down_affine(
    input_size: tuple[int, int],
    bbox_scale: FloatArray,
    bbox_center: FloatArray,
    img: NDArray[np.uint8],
) -> tuple[NDArray[np.uint8], FloatArray]:
    """Crop the bbox area to the model input size, fixing the aspect ratio."""
    w, h = input_size
    warp_size = (int(w), int(h))

    aspect_ratio = w / h
    b_w, b_h = np.hsplit(bbox_scale, [1])
    bbox_scale = np.where(
        b_w > b_h * aspect_ratio,
        np.hstack([b_w, b_w / aspect_ratio]),
        np.hstack([b_h * aspect_ratio, b_h]),
    )

    warp_mat = get_warp_matrix(bbox_center, bbox_scale, 0, output_size=(w, h))
    warped = np.asarray(
        cv2.warpAffine(img, warp_mat, warp_size, flags=cv2.INTER_LINEAR), dtype=np.uint8
    )
    return warped, bbox_scale


def get_simcc_maximum(simcc_x: FloatArray, simcc_y: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Locations (N, K, 2) and values (N, K) of the SimCC maxima; score = mean of axes."""
    n, k, _ = simcc_x.shape
    simcc_x = simcc_x.reshape(n * k, -1)
    simcc_y = simcc_y.reshape(n * k, -1)

    x_locs = np.argmax(simcc_x, axis=1)
    y_locs = np.argmax(simcc_y, axis=1)
    locs = np.stack((x_locs, y_locs), axis=-1).astype(np.float32)
    max_val_x = np.amax(simcc_x, axis=1)
    max_val_y = np.amax(simcc_y, axis=1)

    vals = 0.5 * (max_val_x + max_val_y)
    locs[vals <= 0.0] = -1

    return locs.reshape(n, k, 2), vals.reshape(n, k)
