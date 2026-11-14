# ADR-0015: A training-only dependency group for temporal classifiers

- Status: Accepted
- Date: 2026-11-13

## Context

Issue #15 studies temporal classifiers over tracked poses as a second opinion next to the
fall rules. Training them needs a deep-learning framework, which the approved stack does
not have: deployment runs inference with ONNX Runtime on CPU, and the app image must stay
small and free of training tools.

## Decision

- **PyTorch, in a `train` dependency group** with `onnx` and `onnxscript` for the ONNX
  export. The group is installed only on request (`uv run --group train ...`); the app
  image (`uv sync --no-dev`) and CI (`uv sync`, the `dev` group) never install it.
- **CPU-only wheels** from the PyTorch CPU index, pinned for `torch` alone (an explicit
  uv index). Training runs on the CPU of the development machine.
- **Licenses**: PyTorch BSD-3-Clause, `onnx` Apache-2.0, `onnxscript` MIT: classes the
  project already uses.
- **Deployment stays ONNX Runtime.** A classifier that is ever used by the service is
  exported to ONNX, listed in the model manifest with its SHA-256 and provenance, and
  run by ONNX Runtime. Code under `fall_impact_monitor/training/` may import `torch`;
  nothing the service imports does.
- **Weights** trained on non-commercial or household data are never published
  (ADR-0009); pretrained weights that are openly downloadable may be fine-tuned
  privately, with their terms recorded in `MODELS.md`.

## Alternatives considered

- **TensorFlow / Keras**: no skeleton-model reference implementations to build on here;
  heavier.
- **Training in ONNX Runtime (training APIs)**: immature for these architectures.
- **MMAction2 with mmcv/mmengine**: compiled operators and a large stack for the few
  model definitions needed; the definitions are ported to plain PyTorch instead.
- **A separate repository for training**: splits the data adapter and the evaluation
  from the code that uses them.

## Consequences

- Tests that need PyTorch are skipped where it is not installed (CI); they run locally
  with the `train` group.
- The lock file pins PyTorch, so training is reproducible.
- No change to the app image, its size or the alert path.
