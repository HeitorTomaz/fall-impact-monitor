"""FLASH (HyperMamba) rebuilt for 17 2D keypoints (#15)."""

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from fall_impact_monitor._vendor.flash.hypermamba import coco_hypergraph  # noqa: E402
from fall_impact_monitor.training.models import build_model  # noqa: E402
from fall_impact_monitor.training.train import export_onnx, predict  # noqa: E402
from tests.training.windows_helpers import FEATURES, LENGTH, synthetic  # noqa: E402


def test_the_coco_hypergraph_covers_every_keypoint() -> None:
    incidence, normalised = coco_hypergraph()
    assert incidence.shape[0] == 17 and normalised.shape == (17, 17)
    assert (incidence.sum(dim=1) > 0).all()  # each keypoint is in some hyperedge
    assert torch.allclose(normalised, normalised.T, atol=1e-6)


def test_flash_scores_each_window_and_every_layer_learns() -> None:
    model = build_model("flash", LENGTH, FEATURES)
    x, y = synthetic(4, seed=1)
    assert predict(model, x).shape == (4,)
    model.train()
    torch.nn.functional.binary_cross_entropy_with_logits(
        model(torch.from_numpy(x)), torch.from_numpy(y.astype(np.float32))
    ).backward()
    missing = [n for n, p in model.named_parameters() if p.grad is None or not p.grad.any()]
    assert not missing


def test_flash_ignores_what_padding_rows_hold() -> None:
    model = build_model("flash", LENGTH, FEATURES)
    x, _ = synthetic(2, seed=3)
    x[:, :6] = 0.0
    other = x.copy()
    other[:, :6, :-1] = 5.0  # garbage in rows marked unobserved
    assert np.allclose(predict(model, x), predict(model, other), atol=1e-5)


def test_the_flash_onnx_export_gives_the_same_scores(tmp_path: Path) -> None:
    import onnxruntime

    model = build_model("flash", LENGTH, FEATURES)
    x, _ = synthetic(4, seed=1)
    path = export_onnx(model, tmp_path / "flash.onnx", LENGTH, FEATURES)
    session = onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    (scores,) = session.run(None, {"windows": x})
    assert np.allclose(scores, predict(model, x), atol=1e-4)
