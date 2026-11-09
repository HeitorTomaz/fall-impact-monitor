#!/usr/bin/env python3
# Copyright (c) Megvii Inc. All rights reserved.
# Copyright 2026 The Fall Impact Monitor contributors
# SPDX-License-Identifier: Apache-2.0
#
# Adapted from https://github.com/Megvii-BaseDetection/YOLOX
#   @6ddff4824372906469a7fae2dc3206c7aa4bbaee
#   yolox/utils/demo_utils.py (nms, demo_postprocess)
#   yolox/data/data_augment.py (preproc)
# Changed: type annotations, zip(strict=True), formatting; other helpers removed.

import cv2
import numpy as np
from numpy.typing import NDArray


def nms(boxes: NDArray[np.float32], scores: NDArray[np.float32], nms_thr: float) -> list[int]:
    """Single class NMS implemented in Numpy."""
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep: list[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= nms_thr)[0]
        order = order[inds + 1]

    return keep


def demo_postprocess(
    outputs: NDArray[np.float32], img_size: tuple[int, int], p6: bool = False
) -> NDArray[np.float32]:
    grids = []
    expanded_strides = []
    strides = [8, 16, 32] if not p6 else [8, 16, 32, 64]

    hsizes = [img_size[0] // stride for stride in strides]
    wsizes = [img_size[1] // stride for stride in strides]

    for hsize, wsize, stride in zip(hsizes, wsizes, strides, strict=True):
        xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
        grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
        grids.append(grid)
        shape = grid.shape[:2]
        expanded_strides.append(np.full((*shape, 1), stride))

    grid_all = np.concatenate(grids, 1)
    stride_all = np.concatenate(expanded_strides, 1)
    outputs[..., :2] = (outputs[..., :2] + grid_all) * stride_all
    outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * stride_all

    return outputs


def preproc(
    img: NDArray[np.uint8], input_size: tuple[int, int], swap: tuple[int, int, int] = (2, 0, 1)
) -> tuple[NDArray[np.float32], float]:
    padded_img: NDArray[np.uint8]
    if len(img.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114
    else:
        padded_img = np.ones(input_size, dtype=np.uint8) * 114

    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.uint8)
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

    padded_img = padded_img.transpose(swap)
    blob = np.ascontiguousarray(padded_img, dtype=np.float32)
    return blob, r
