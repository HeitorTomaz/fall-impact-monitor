"""`fim train`: train a temporal classifier on ``fim windows`` files (#15).

Splits follow ``datasets.toml`` (by person or event): fit on train, keep the epoch with
the best validation average precision, score the test split, export ONNX. Needs the
``train`` dependency group (ADR-0015). Models trained on non-commercial data are never
published (ADR-0009).
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from fall_impact_monitor.evaluation.datasets import load_dataset_manifest


def add_parser(subparsers: "argparse._SubParsersAction[argparse.ArgumentParser]") -> None:
    parser = subparsers.add_parser("train", help="train a temporal classifier on pose windows")
    parser.add_argument("--windows", type=Path, nargs="+", required=True, help="fim windows files")
    parser.add_argument("--model", default="tcn", help="architecture (tcn)")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--manifest", type=Path, default=Path("datasets.toml"))
    parser.add_argument(
        "--pretrained", help="published weights to start from (training.pretrained)"
    )
    parser.add_argument("--models-dir", type=Path, default=Path("models"), help="where they are")
    parser.add_argument("--out-dir", type=Path, required=True, help="model.onnx and metrics.json")
    parser.set_defaults(handler=run)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as data:
        for chunk in iter(lambda: data.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(args: argparse.Namespace) -> int:
    try:
        from fall_impact_monitor.training.models import build_model
        from fall_impact_monitor.training.train import export_onnx, fit, predict
    except ImportError:
        print("train: needs the train dependency group: uv run --group train fim train ...")
        return 2
    from fall_impact_monitor.training.data import average_precision, load_windows, split_rows
    from fall_impact_monitor.training.pretrained import PRETRAINED, load_pretrained

    start = PRETRAINED.get(args.pretrained) if args.pretrained else None
    if args.pretrained and (start is None or start.model != args.model):
        print(f"train: no published weights {args.pretrained!r} for model {args.model!r}")
        return 2
    weights = args.models_dir / f"{args.pretrained}.pth" if start else None
    if start and weights and not weights.exists():
        print(
            f"train: download {start.url} to {weights} first "
            f"(it is checked against sha256 {start.sha256})"
        )
        return 2

    windows = load_windows(args.windows)
    splits = split_rows(windows, load_dataset_manifest(args.manifest))
    train, validation, test = (windows.subset(splits[s]) for s in ("train", "validation", "test"))
    _, length, features = windows.features.shape
    model = build_model(args.model, length, features, seed=args.seed)
    if start and weights:
        from fall_impact_monitor.training.models import STGCNPP, PoseC3D

        assert isinstance(model, STGCNPP | PoseC3D)  # PRETRAINED entries name the model
        load_pretrained(model, weights, start.sha256)
    result = fit(
        model,
        (train.features, train.labels),
        (validation.features, validation.labels),
        epochs=args.epochs,
        seed=args.seed,
    )
    scores = predict(model, test.features)
    by_dataset = {
        str(d): average_precision(test.labels[test.datasets == d], scores[test.datasets == d])
        for d in np.unique(test.datasets)
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    export_onnx(model, args.out_dir / "model.onnx", length, features)
    metrics = {
        "model": args.model,
        "seed": args.seed,
        "epochs": args.epochs,
        "best_epoch": result.best_epoch,
        "validation_ap": result.validation_ap,
        "validation_ap_history": result.history,
        "test_ap": average_precision(test.labels, scores),
        "test_ap_by_dataset": by_dataset,
        "windows": {
            "train": len(train.labels),
            "validation": len(validation.labels),
            "test": len(test.labels),
        },
        "falls": {
            "train": int(train.labels.sum()),
            "validation": int(validation.labels.sum()),
            "test": int(test.labels.sum()),
        },
        "inputs": [{"file": p.name, "sha256": _sha256(p)} for p in args.windows],
        "perception": windows.perception,
        "pretrained": {"name": args.pretrained, "sha256": start.sha256} if start else None,
    }
    (args.out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(
        f"train: {args.model}: validation AP {result.validation_ap:.3f} (epoch "
        f"{result.best_epoch}), test AP {metrics['test_ap']:.3f} -> {args.out_dir}"
    )
    return 0
