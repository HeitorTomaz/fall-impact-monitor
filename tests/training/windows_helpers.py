"""Synthetic pose windows for the training tests (no PyTorch needed)."""

import numpy as np

LENGTH, FEATURES = 16, 53
HIP_Y = 11 * 3 + 1


def synthetic(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Falls: the hips drop by half a height over the last rows; others stand still."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 0.02, (n, LENGTH, FEATURES)).astype(np.float32)
    x[:, :, -1] = 1.0
    x[:, 1:, -2] = 0.125
    y = (rng.random(n) < 0.3).astype(np.int8)
    drop = np.clip(np.arange(LENGTH) - (LENGTH - 6), 0, None) * 0.1
    x[y == 1, :, HIP_Y] += drop.astype(np.float32)
    return x, y
