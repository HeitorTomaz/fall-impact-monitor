# Temporal classifiers (study, #15)

Experimental. Temporal models over tracked poses are studied as a **second opinion**
next to the fall rules: they are measured against the rules alone on the same held-out
events, they never suppress an alert, and none is enabled by default. A change to the
alert policy would need measured evidence and an ADR.

## Data

`fim windows` turns the analyses that `fim eval` saved into labelled pose windows:

```bash
docker compose --profile dev run --rm dev uv run --frozen \
  fim windows --dataset caucafall --root <dataset folder> --out-dir <fim eval --out-dir> \
  --output artifacts/windows/caucafall.npz
```

- **Window** (`analysis/pose_window.py`): a tracked person's last 16 analysed poses
  (about 2 s at 8 analysed frames per second), oldest first. Each row holds the 17
  COCO keypoints (x, y, confidence), the real time since the previous row and an
  `observed` flag. Only observed frames are used, at their irregular times; a short
  track is padded with zero rows marked unobserved. Nothing is interpolated.
- **Normalisation**: one reference per window, so motion stays visible: coordinates are
  relative to the hip midpoint of the window's first pose and divided by the person's
  tallest box in the window. Image y grows downwards.
- **Labels**: 1 when the window's time span overlaps a labelled fall, 0 when it does
  not; windows ending within 1 s after a fall (the person lying right after it) are left
  out.
- **Groups**: the person (GMDCSA24 and CAUCAFall subjects), the sequence (UR Fall, which
  has no subject labels) or the scenario (MCFD, whose eight views of one event share a
  group). Splits keep a group on one side, so no person or event is in both training
  and test data.
- The file records the perception configuration of the analyses; windows from
  analyses made with different configurations are refused.

The `.npz` files hold pose traces of the people in the datasets: keep them local
(`artifacts/` is git-ignored) and never publish them.

## Training

```bash
docker compose --profile dev run --rm dev uv run --frozen --group train \
  fim train --windows artifacts/windows/*.npz --model tcn --out-dir artifacts/classifiers/tcn
```

- PyTorch comes from the optional `train` dependency group (CPU wheels; ADR-0015). The
  app image and CI never install it; the tests that need it are skipped there.
- Splits follow `datasets.toml`: train on `train`, keep the epoch with the best
  window-level average precision on `validation`, score `test` (overall and per
  dataset). Falls are weighted by their rarity in the loss.
- Outputs: `model.onnx` (input `windows`, any batch; output `scores`, fall
  probabilities) and `metrics.json` (seed, epochs, splits, scores, SHA-256 of the input
  files, perception configuration). Training is reproducible from the seed.
- Window-level scores rank windows; event-level recall, false alerts per hour and
  latency against the rules come with the shadow-mode integration (replayed with
  `fim eval --replay`).
- `--pretrained <name>` starts from published weights listed in
  `training/pretrained.py`: the file must be in `--models-dir` (default `models/`,
  git-ignored), it is checked against its SHA-256 and read with PyTorch's weights-only
  loader. Only the backbone is loaded; the fall output is new.

### Candidates

| Model | Status |
|---|---|
| `tcn`: small dilated causal TCN (cost and quality baseline, written for this project) | Implemented |
| `stgcnpp`: ST-GCN++ (MMAction2 configuration, ported to plain PyTorch in `_vendor/mmaction2`, Apache-2.0), from scratch or fine-tuned with `--pretrained stgcnpp_ntu60_2d_joint` (NTU-60 2D weights, see `MODELS.md`; download them to `models/` first) | Implemented |
| TCN + Transformer encoder (TCNTE; reimplemented from the published description, no code available) | Planned |
| PoseC3D (MMAction2, Apache-2.0), reduced | Planned |
| FLASH (MIT), rebuilt for 17 2D keypoints | Planned |
