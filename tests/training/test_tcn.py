"""Training and exporting the small TCN baseline (#15). Needs the train group."""

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from fall_impact_monitor.training.data import average_precision  # noqa: E402
from fall_impact_monitor.training.models import build_model  # noqa: E402
from fall_impact_monitor.training.train import export_onnx, fit, predict  # noqa: E402
from tests.training.windows_helpers import FEATURES, LENGTH, synthetic  # noqa: E402


def test_the_baseline_learns_a_drop() -> None:
    model = build_model("tcn", LENGTH, FEATURES)
    x, y = synthetic(400, seed=1)
    vx, vy = synthetic(200, seed=2)
    result = fit(model, (x, y), (vx, vy), epochs=15, seed=0)
    assert result.validation_ap > 0.95
    tx, ty = synthetic(200, seed=3)
    scores = predict(model, tx)
    assert scores.shape == (200,) and scores.min() >= 0.0 and scores.max() <= 1.0
    assert average_precision(ty, scores) > 0.95  # ranking; thresholds come from validation


def test_training_is_reproducible() -> None:
    x, y = synthetic(200, seed=1)
    runs = []
    for _ in range(2):
        model = build_model("tcn", LENGTH, FEATURES, seed=7)
        fit(model, (x, y), (x, y), epochs=3, seed=7)
        runs.append(predict(model, x[:10]))
    assert np.allclose(runs[0], runs[1])


def test_the_onnx_export_gives_the_same_scores(tmp_path: Path) -> None:
    import onnxruntime

    model = build_model("tcn", LENGTH, FEATURES)
    x, y = synthetic(64, seed=1)
    fit(model, (x, y), (x, y), epochs=2, seed=0)
    path = export_onnx(model, tmp_path / "tcn.onnx", LENGTH, FEATURES)
    session = onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    (onnx_scores,) = session.run(None, {"windows": x})
    assert np.allclose(onnx_scores, predict(model, x), atol=1e-4)
