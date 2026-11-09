# ADR-0009: Model and dataset policy

- Status: Accepted
- Date: 2026-11-08
- Supersedes: the strict dataset and weight approval policy of architecture v0.1 and issue #2

## Context

Architecture v0.1 required every dataset and pretrained weight to carry verified permissive terms for commercial use. Under that policy only two small staged fall datasets qualified, and the best CPU pose models were excluded. The purpose of the system is residential safety, and the maintainer prioritizes detection quality for installations. At the same time, the public repository is Apache-2.0 and must remain safely redistributable by anyone.

## Decision

**Installations choose models and data for detection quality.**

- Any published, **openly downloadable** model or dataset may be used privately by the installer for evaluation, calibration and training, chosen by measured benchmark results.
- Datasets that require registration or request forms are out of scope.
- Terms of every model and dataset are recorded (`MODELS.md`, `docs/DATASETS.md`): source, licence, training data, permitted use, SHA-256, conversions.
- Nothing is redistributed by the project, and every source is cited.

**The repository stays redistributable under Apache-2.0.**

- No GPL, AGPL or unlicensed code is copied into the repository; such projects may inform ideas only.
- No weights, datasets or private media are committed.
- Permissively licensed code may be vendored under `src/fall_impact_monitor/_vendor/<origin>/`, keeping original copyright headers, adding SPDX identifiers and a "modified" note, and recording it in `THIRD_PARTY_NOTICES.md`; full licence texts go in `LICENSES/`.
- Restrictively licensed models (for example AGPL detector weights) are used only as downloaded artifacts (exported ONNX), never as imported packages.

**Trained weights.**

- Weights trained on household recordings, or on non-commercial or research-only data, are never published.
- Weights trained only on redistributable datasets may be published later with attribution, after review.

**Dependencies** are installed packages, not copies; their licences are checked when added. Packages whose licence is incompatible with Apache-2.0 distribution (e.g. GPL Python libraries imported by the core) are avoided.

## Alternatives considered

- **Keep the strict v0.1 policy**: legally clean for commercial adopters, but materially worse detection; rejected.
- **Bundle the best models in the repository or images**: would make the repository non-redistributable; rejected.
- **Request permission for every research dataset**: not pursued; such datasets are simply excluded when not openly downloadable.

## Consequences

- The default model selection may carry non-commercial or unclear terms; commercial adopters must pick alternatives, and the documentation says so.
- Evaluation can use a much larger corpus of non-commercial fall datasets privately.
- Licence review shifts from gatekeeping to documentation, but repository hygiene remains a hard rule checked in every pull request.

## References

- https://www.apache.org/licenses/LICENSE-2.0
- https://www.apache.org/legal/resolved.html
- https://reuse.software/spec-3.3/
