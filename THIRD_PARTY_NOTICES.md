# Third-party notices

Fall Impact Monitor is licensed under the Apache License 2.0 (see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE)). This file lists third-party source code that is copied or adapted into this repository.

Included code is listed under [Included](#included); the entries under [Planned adaptations](#planned-adaptations) are not yet included.

Dependencies installed by the package manager or run as container images are not copied into this repository and are not listed here. Model weights are covered by [`MODELS.md`](MODELS.md), datasets by [`docs/DATASETS.md`](docs/DATASETS.md), and projects that informed the design without code being copied by [`CREDITS.md`](CREDITS.md).

## Convention

- Only code under a license compatible with Apache-2.0 is copied (for example Apache-2.0, MIT, BSD, ISC). GPL, AGPL and unlicensed code is never copied.
- Copied code lives in `src/fall_impact_monitor/_vendor/<origin>/`, one directory per origin.
- Each vendored file keeps its original copyright and license header unchanged, and adds:
  - an SPDX line, for example `# SPDX-License-Identifier: MIT`;
  - a provenance line: `# Adapted from <url>@<commit>; changed: <summary of changes>`.
- When the first file under a given license is vendored, the full license text is added to `LICENSES/`, named by its SPDX identifier (for example `LICENSES/MIT.txt`), following the [REUSE](https://reuse.software/) layout.
- Each vendored origin gets an entry in this file, in the same pull request that adds the code.
- If an origin ships its own `NOTICE` file, its required attribution is carried into the project [`NOTICE`](NOTICE).

## Entry template

```markdown
### <component name>

- Upstream: <repository URL>
- Version: <tag or commit SHA>
- License: <SPDX identifier> (full text in `LICENSES/<SPDX identifier>.txt`)
- Copyright: <copyright lines exactly as in the upstream files>
- Upstream files: <paths in the upstream repository>
- Vendored as: `src/fall_impact_monitor/_vendor/<origin>/<file>`
- Changes: <what was changed and why>
- Used by: <project modules that use it>
```

## Included

### YOLOX pre- and post-processing

- Upstream: <https://github.com/Megvii-BaseDetection/YOLOX>
- Version: 6ddff4824372906469a7fae2dc3206c7aa4bbaee
- License: Apache-2.0 (full text in `LICENSES/Apache-2.0.txt`)
- Copyright: Copyright (c) Megvii Inc. All rights reserved.
- Upstream files: `yolox/utils/demo_utils.py` (`nms`, `demo_postprocess`), `yolox/data/data_augment.py` (`preproc`)
- Vendored as: `src/fall_impact_monitor/_vendor/yolox/utils.py`
- Changes: type annotations, `zip(strict=True)`, formatting; unused helpers removed.
- Used by: `fall_impact_monitor.perception.detector`

### rtmlib pre- and post-processing

- Upstream: <https://github.com/Tau-J/rtmlib>
- Version: 5ea82df885d6f2a66923f1de2d15c799af0ed732
- License: Apache-2.0 (full text in `LICENSES/Apache-2.0.txt`)
- Copyright: Copyright 2018-2020 Open-MMLab. (as stated in the upstream `LICENSE`; the upstream files carry no header)
- Upstream files: `rtmlib/tools/pose_estimation/pre_processings.py` (`bbox_xyxy2cs`, `_rotate_point`, `_get_3rd_point`, `get_warp_matrix`, `top_down_affine`), `rtmlib/tools/pose_estimation/post_processings.py` (`get_simcc_maximum`)
- Vendored as: `src/fall_impact_monitor/_vendor/rtmlib/processing.py`
- Changes: type annotations, float32 arrays passed directly to `getAffineTransform`, the "mean" score combination inlined, shorter docstrings, formatting; other helpers removed.
- Used by: `fall_impact_monitor.perception.pose`

## Planned adaptations

Planned, not yet included. Upstream licenses were read from each repository's `LICENSE` file during design and must be re-checked at the pinned commit when the code is vendored. Paths refer to the upstream repositories.

| Component | Upstream | Upstream files | License | Intended use | Status |
|---|---|---|---|---|---|
| edgefallkit fall state machine | <https://github.com/suharvest/edgefallkit> | `platforms/rknn/fall_core.py` and its scenario tests | Apache-2.0 | Base of the two-level fall state machine (`normal → suspected → fallen → recovering → normal`) and its configuration. The project's weights and temporal classifier are not used | Planned, not yet included |
| ambianic fall-detection | <https://github.com/ambianic/fall-detection> | `src/pipeline/fall_detect.py` | Apache-2.0 | Retrying pose estimation on a crop rotated by ±90° for people lying down; angle helpers | Planned, not yet included |
| HumanFallDetection | <https://github.com/taufeeque9/HumanFallDetection> | `vis/inv_pendulum.py` | MIT | Head and neck rotational energy, inverted-pendulum model, bounding-box ratio change rate (adapted to NumPy 2) | Planned, not yet included |
| YOLOX pre- and post-processing | <https://github.com/Megvii-BaseDetection/YOLOX> | `yolox/data/data_augment.py` (`preproc`), `yolox/utils/demo_utils.py` (`nms`, `demo_postprocess`) | Apache-2.0 | Running YOLOX ONNX models without installing the `yolox` package | Included (see above) |
| rtmlib pre- and post-processing | <https://github.com/Tau-J/rtmlib> | `rtmlib/tools/pose_estimation/pre_processings.py`, `post_processings.py` (`top_down_affine`, `get_simcc_maximum`) | Apache-2.0 | Running RTMPose and RTMW ONNX models without depending on `rtmlib` | Included (see above) |
| One Euro filter (one of two options) | <https://github.com/casiez/OneEuroFilter> (author's implementation), or <https://github.com/open-mmlab/mmpose/tree/0.x/mmpose/core/post_processing/temporal_filters> (mmpose 0.x) | `python/OneEuroFilter/OneEuroFilter.py`, or mmpose 0.x `temporal_filters/one_euro_filter.py` | BSD-3-Clause, or Apache-2.0 with MIT-licensed portions | Smoothing keypoints before kinematics | Planned, not yet included |
| caregiving-eye RTSP helpers | <https://github.com/JuneYaooo/caregiving-eye> | `mibutler/abilities/audio.py`, `mibutler/brain/frames.py` | MIT | Reference for the RTSP audio reader (silence gate, restart on stall) and latest-frame reading with reconnection backoff | Planned, not yet included |
| librosa onset detection | <https://github.com/librosa/librosa> | `librosa/onset.py` (`onset_strength`), `librosa/util/utils.py` (`peak_pick`), minimal subset | ISC | Low-cost onset filter for impact sounds | Planned, not yet included |
| PANNs model definition | <https://github.com/qiuqiangkong/audioset_tagging_cnn> | `pytorch/models.py` (`Cnn14_8k`) | MIT | Model definition used to export the audio tagger to ONNX at build time. The weights are listed in [`MODELS.md`](MODELS.md) | Planned, not yet included |
| Frigate FFmpeg presets (fallback only) | <https://github.com/blakeblackshear/frigate> | `frigate/ffmpeg_presets.py` | MIT | Segment recording with FFmpeg, only if MediaMTX recording fails the long-run test | Planned, not yet included |
| raptor-test RTSP checks | <https://github.com/gtxaspec/raptor-test> | TBD | MIT | RTSP diagnostics: RTP timestamp monotonicity, audio drift, RTCP sender reports | Planned, not yet included |
