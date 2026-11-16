"""PoseC3D (MMAction2 SlowOnly-R50 over keypoint heatmaps), reduced (#15)."""

import hashlib
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from fall_impact_monitor.training.models import (  # noqa: E402
    HEATMAP_EXTENT,
    HEATMAP_SIZE,
    build_model,
    keypoint_heatmaps,
)
from fall_impact_monitor.training.pretrained import load_pretrained  # noqa: E402
from fall_impact_monitor.training.train import export_onnx, predict  # noqa: E402
from tests.training.windows_helpers import FEATURES, LENGTH, synthetic  # noqa: E402


def test_a_heatmap_peaks_at_its_keypoint_and_scales_with_confidence() -> None:
    windows = torch.zeros(1, 2, FEATURES)
    windows[:, :, -1] = 1.0  # observed
    half_pixel = 0.5 * 2 * HEATMAP_EXTENT / (HEATMAP_SIZE - 1)  # onto pixel (16, 16)
    windows[0, 1, 0:3] = torch.tensor([half_pixel, half_pixel, 0.5])  # nose, score 0.5
    maps = keypoint_heatmaps(windows)
    assert maps.shape == (1, 17, 2, HEATMAP_SIZE, HEATMAP_SIZE)
    nose = maps[0, 0, 1]
    assert divmod(int(nose.argmax()), HEATMAP_SIZE) == (HEATMAP_SIZE // 2, HEATMAP_SIZE // 2)
    assert float(nose.max()) == pytest.approx(0.5, abs=1e-5)


def test_unobserved_rows_give_empty_heatmaps() -> None:
    x, _ = synthetic(1, seed=0)
    x[0, :4, -1] = 0.0
    maps = keypoint_heatmaps(torch.from_numpy(x))
    assert float(maps[0, :, :4].abs().sum()) == 0.0


def test_posec3d_scores_each_window_and_every_layer_learns() -> None:
    model = build_model("posec3d", LENGTH, FEATURES)
    x, y = synthetic(2, seed=1)
    assert predict(model, x).shape == (2,)
    model.train()
    torch.nn.functional.binary_cross_entropy_with_logits(
        model(torch.from_numpy(x)), torch.from_numpy(y.astype(np.float32))
    ).backward()
    missing = [n for n, p in model.named_parameters() if p.grad is None or not p.grad.any()]
    assert not missing


def test_published_posec3d_backbone_weights_load(tmp_path: Path) -> None:
    source = build_model("posec3d", LENGTH, FEATURES, seed=1)
    state = {f"backbone.{k}": v for k, v in source.backbone.state_dict().items()}
    state["cls_head.fc_cls.weight"] = torch.zeros(60, 512)
    state["cls_head.fc_cls.bias"] = torch.zeros(60)
    path = tmp_path / "ckpt.pth"
    torch.save({"state_dict": state}, path)
    model = build_model("posec3d", LENGTH, FEATURES, seed=2)
    load_pretrained(model, path, hashlib.sha256(path.read_bytes()).hexdigest())
    for name, value in source.backbone.state_dict().items():
        assert torch.equal(model.backbone.state_dict()[name], value), name


def test_the_posec3d_onnx_export_gives_the_same_scores(tmp_path: Path) -> None:
    import onnxruntime

    model = build_model("posec3d", LENGTH, FEATURES)
    x, _ = synthetic(2, seed=1)
    path = export_onnx(model, tmp_path / "posec3d.onnx", LENGTH, FEATURES)
    session = onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    (scores,) = session.run(None, {"windows": x})
    assert np.allclose(scores, predict(model, x), atol=1e-4)
