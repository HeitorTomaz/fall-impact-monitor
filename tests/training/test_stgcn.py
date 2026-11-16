"""ST-GCN++ from MMAction2, from scratch and from published weights (#15)."""

import hashlib
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from fall_impact_monitor.training.models import build_model  # noqa: E402
from fall_impact_monitor.training.pretrained import load_pretrained  # noqa: E402
from fall_impact_monitor.training.train import export_onnx, predict  # noqa: E402
from tests.training.windows_helpers import FEATURES, LENGTH, synthetic  # noqa: E402


def test_stgcnpp_scores_each_window() -> None:
    model = build_model("stgcnpp", LENGTH, FEATURES)
    x, _ = synthetic(4, seed=0)
    assert predict(model, x).shape == (4,)


def test_every_stgcnpp_layer_gets_a_gradient() -> None:
    """The port is wired end to end (whether it learns falls is measured on real data)."""
    model = build_model("stgcnpp", LENGTH, FEATURES)
    x, y = synthetic(8, seed=1)
    logits = model(torch.from_numpy(x))
    torch.nn.functional.binary_cross_entropy_with_logits(
        logits, torch.from_numpy(y.astype(np.float32))
    ).backward()
    missing = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and (parameter.grad is None or not parameter.grad.any())
    ]
    assert not missing


def checkpoint(path: Path, *, with_backbone: bool = True) -> str:
    """An MMAction2-style checkpoint: backbone weights and a 60-class head."""
    source = build_model("stgcnpp", LENGTH, FEATURES, seed=1)
    state = {f"backbone.{k}": v for k, v in source.backbone.state_dict().items()}
    if not with_backbone:
        state = {}
    state["cls_head.fc.weight"] = torch.zeros(60, 256)
    state["cls_head.fc.bias"] = torch.zeros(60)
    torch.save({"meta": {"epoch": 80}, "state_dict": state}, path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_published_backbone_weights_are_loaded_and_the_head_is_new(tmp_path: Path) -> None:
    path = tmp_path / "ckpt.pth"
    digest = checkpoint(path)
    model = build_model("stgcnpp", LENGTH, FEATURES, seed=2)
    head_before = model.cls_head.fc.weight.detach().clone()
    load_pretrained(model, path, digest)
    source = build_model("stgcnpp", LENGTH, FEATURES, seed=1)
    for name, value in source.backbone.state_dict().items():
        assert torch.equal(model.backbone.state_dict()[name], value), name
    assert torch.equal(model.cls_head.fc.weight, head_before)  # one fall output, not 60


def test_a_checkpoint_that_does_not_match_its_hash_is_not_loaded(tmp_path: Path) -> None:
    path = tmp_path / "ckpt.pth"
    checkpoint(path)
    with pytest.raises(ValueError, match="sha256"):
        load_pretrained(build_model("stgcnpp", LENGTH, FEATURES), path, "0" * 64)


def test_a_checkpoint_without_the_backbone_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "ckpt.pth"
    digest = checkpoint(path, with_backbone=False)
    with pytest.raises(ValueError, match="backbone"):
        load_pretrained(build_model("stgcnpp", LENGTH, FEATURES), path, digest)


def test_the_stgcnpp_onnx_export_gives_the_same_scores(tmp_path: Path) -> None:
    import onnxruntime

    model = build_model("stgcnpp", LENGTH, FEATURES)
    x, _ = synthetic(8, seed=1)
    path = export_onnx(model, tmp_path / "stgcnpp.onnx", LENGTH, FEATURES)
    session = onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    (scores,) = session.run(None, {"windows": x})
    assert np.allclose(scores, predict(model, x), atol=1e-4)
