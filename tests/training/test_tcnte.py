"""TCN + Transformer encoder (TCNTE), reimplemented from its published description (#15)."""

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from fall_impact_monitor.training.models import build_model  # noqa: E402
from fall_impact_monitor.training.train import export_onnx, predict  # noqa: E402
from tests.training.windows_helpers import FEATURES, LENGTH, synthetic  # noqa: E402


def test_tcnte_scores_each_window_and_every_layer_learns() -> None:
    model = build_model("tcnte", LENGTH, FEATURES)
    x, y = synthetic(8, seed=1)
    assert predict(model, x).shape == (8,)
    model.train()
    torch.nn.functional.binary_cross_entropy_with_logits(
        model(torch.from_numpy(x)), torch.from_numpy(y.astype(np.float32))
    ).backward()
    assert all(p.grad is not None and p.grad.any() for p in model.parameters())


def test_padding_rows_do_not_change_the_score() -> None:
    """Unobserved (padding) rows are masked in attention: a short track scores the same
    whatever its padding holds."""
    model = build_model("tcnte", LENGTH, FEATURES)
    x, _ = synthetic(2, seed=3)
    x[:, :6, -1] = 0.0  # the first six rows are padding
    x[:, :6, :-1] = 0.0
    other = x.copy()
    other[:, :6, :-1] = 5.0  # garbage in the padding
    assert np.allclose(predict(model, x), predict(model, other), atol=1e-5)


def test_the_tcnte_onnx_export_gives_the_same_scores(tmp_path: Path) -> None:
    import onnxruntime

    model = build_model("tcnte", LENGTH, FEATURES)
    x, _ = synthetic(8, seed=1)
    x[:3, :5, -1] = 0.0  # some padded windows
    path = export_onnx(model, tmp_path / "tcnte.onnx", LENGTH, FEATURES)
    session = onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    (scores,) = session.run(None, {"windows": x})
    assert np.allclose(scores, predict(model, x), atol=1e-4)
