# Models

This file records every model weight that Fall Impact Monitor may use: where it comes from, its license, the data it was trained on, its checksum and how it is converted.

**Status:** the defaults are selected (see [Selected models](#selected-models)); the other entries remain candidates. Person detector and pose models are chosen by benchmark (detection quality on people who are far away, lying down or in unusual postures, within the CPU budget).

## Policy

- **No weights in the repository.** Weights are never committed to Git and never redistributed by this project, including inside container images (images are built locally and never published).
- **Downloaded at build time from the original source.** The image build downloads each selected weight from the URL published by its authors, as listed in the model manifest (source URL, SHA-256, license, provenance). Nothing is downloaded at runtime, so the system works offline.
- **SHA-256 verified.** The build fails if a downloaded file does not match the SHA-256 recorded in the manifest. Downloaded files are treated as untrusted; untrusted pickles are never loaded.
- **Terms recorded.** Installations choose models for detection quality. Any published, openly downloadable model may be used privately by the installer. Its license and the terms of its training data are recorded here, and its authors are cited. Models whose download requires registration or a request form are out of scope.
- **Conversions recorded.** Any conversion (for example PyTorch to ONNX, quantization) is described in the table, together with where it runs.
- **Trained weights are never published.** Weights trained or fine-tuned on household recordings, or on non-commercial or research-only data, are never published.
- Adopters with commercial use in mind must review each license and choose permissively licensed alternatives where needed.

## Candidate models

"TBD" means not yet determined. Where the exact file URL is not yet fixed, the source column links to the project page; the exact URL is recorded in the manifest when a model is selected. "Not stated separately" means the authors publish no license specific to the weights; the repository license is shown for reference.

### Person detection

| Model | Purpose | Source URL | License of weights | Training data and its terms | Status | SHA-256 | Conversions |
|---|---|---|---|---|---|---|---|
| YOLOX-Tiny (Megvii) | Person detection | <https://github.com/Megvii-BaseDetection/YOLOX> | Not stated separately (repository: Apache-2.0) | COCO 2017 (annotations CC BY 4.0; images subject to Flickr terms) | Candidate | TBD | TBD |
| YOLOX-S (Megvii) | Person detection | <https://github.com/Megvii-BaseDetection/YOLOX> | Not stated separately (repository: Apache-2.0) | COCO 2017 (annotations CC BY 4.0; images subject to Flickr terms) | Candidate | TBD | TBD |
| YOLOX-HumanArt (tiny/s/m, as used by rtmlib) | Person detection, including unusual postures | <https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose> (also used by <https://github.com/Tau-J/rtmlib>) | Not stated separately (repository: Apache-2.0) | Human-Art + COCO. Human-Art is for non-commercial use and access is granted through a request form | Candidate | TBD | TBD (published as ONNX) |
| YOLO11 (Ultralytics) | Person detection | <https://github.com/ultralytics/ultralytics> | AGPL-3.0 (or an Ultralytics commercial license) | COCO 2017 (annotations CC BY 4.0; images subject to Flickr terms) | Candidate | TBD | Exported to ONNX by the installer, outside this repository and the image; the `ultralytics` package is never part of either. Procedure TBD |
| YOLO26 (Ultralytics) | Person detection | <https://github.com/ultralytics/ultralytics> | AGPL-3.0 (or an Ultralytics commercial license) | COCO 2017 (annotations CC BY 4.0; images subject to Flickr terms) | Candidate | TBD | As for YOLO11 |
| RTMDet (OpenMMLab) | Person detection | <https://github.com/open-mmlab/mmdetection/tree/main/configs/rtmdet> | Not stated separately (repository: Apache-2.0) | COCO 2017 for general models; data of person-only variants TBD | Candidate | TBD | TBD |
| D-FINE | Person detection | <https://github.com/Peterande/D-FINE> | Not stated separately (repository: Apache-2.0) | COCO 2017; some checkpoints are also pre-trained on Objects365 (terms TBD) | Candidate | TBD | TBD |

### Pose estimation

| Model | Purpose | Source URL | License of weights | Training data and its terms | Status | SHA-256 | Conversions |
|---|---|---|---|---|---|---|---|
| RTMPose-s / RTMPose-m, body7 variants | Body keypoints (COCO-17), top-down on person crops | <https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose> | Not stated separately (repository: Apache-2.0) | Body7: AI Challenger, COCO, CrowdPose, MPII, sub-JHMDB, Halpe, PoseTrack18. Mixed terms, some research-only or non-commercial | Candidate | TBD | TBD (published as ONNX) |
| RTMW | Whole-body keypoints including hands, confirmation stage only (for example "hand on head") | <https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose> | Not stated separately (repository: Apache-2.0) | Cocktail14: the Body7 datasets plus COCO-WholeBody, UBody, Human-Art, WFLW, 300W, COFW, LaPa, InterHand. Mixed terms, some research-only or non-commercial | Candidate | TBD | TBD (published as ONNX) |

### Audio

| Model | Purpose | Source URL | License of weights | Training data and its terms | Status | SHA-256 | Conversions |
|---|---|---|---|---|---|---|---|
| PANNs Cnn14_8k | Audio tagging of impact-like sounds at 8 kHz | <https://zenodo.org/records/3987831> (Zenodo 3987831); model code: <https://github.com/qiuqiangkong/audioset_tagging_cnn> (MIT) | CC BY 4.0 | AudioSet (label annotations CC BY 4.0; ontology CC BY-SA 4.0; audio from third-party online videos) | Candidate | TBD | PyTorch to ONNX in a dedicated build stage; PyTorch is used only in that stage. Exact procedure TBD |

PANNs attribution (CC BY 4.0): the selected entry will carry the record title, creators, source URL and license exactly as stated on the Zenodo record, plus a note that the weights were converted to ONNX.

The AudioSet class list (label names) is licensed CC BY-SA 4.0. It is kept in a separate file with its own license notice and is not mixed into Apache-2.0 source files.

### Depth (post-MVP)

| Model | Purpose | Source URL | License of weights | Training data and its terms | Status | SHA-256 | Conversions |
|---|---|---|---|---|---|---|---|
| Depth Anything V2 Small | Monocular depth for head-impact candidates | <https://github.com/DepthAnything/Depth-Anything-V2> | Apache-2.0 | Synthetic images with depth labels plus large-scale pseudo-labelled real images (terms TBD) | Candidate (post-MVP) | TBD | TBD |
| Depth Anything V2 Base / Large | Monocular depth for head-impact candidates | <https://github.com/DepthAnything/Depth-Anything-V2> | CC BY-NC 4.0 (non-commercial; private use only) | Synthetic images with depth labels plus large-scale pseudo-labelled real images (terms TBD) | Candidate (post-MVP) | TBD | TBD |

## Selected models

All models in `models.toml` are verified by SHA-256 of the download and of the ONNX file. Exports are used as published (no conversion).

| Name (`models.toml`) | Role | Source | Download SHA-256 | ONNX SHA-256 | Weights license | Training data |
|---|---|---|---|---|---|---|
| `yolox_m_humanart` (default detector) | Person detection, incl. lying people | <https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_m_8xb8-300e_humanart-c2c7a14a.zip> | `a000224f…adde` | `3dea6513…2f5f23` | Not stated separately (repository: Apache-2.0) | Human-Art (non-commercial terms) |
| `yolox_s` (permissive alternative) | Person detection; bed and sofa detection for the safe zones suggested by `fim zones` (fetched by default) | <https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.onnx> | `c5c2d13e…998063` | same file | Apache-2.0 (repository) | COCO train2017 |
| `yolox_tiny` | Person detection, weak CPUs | <https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_tiny.onnx> | `427cc366…0b0f7` | same file | Apache-2.0 (repository) | COCO train2017 |
| `rtmpose_s` (default pose) | COCO-17 pose | <https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-s_simcc-body7_pt-body7_420e-256x192-acd4a1ef_20230504.zip> | `7673922e…0d802` | `9aeb635b…f47d` | Not stated separately (repository: Apache-2.0) | body7 (AI Challenger, COCO, CrowdPose, MPII, sub-JHMDB, Halpe, PoseTrack18) |
| `rtmpose_m` | COCO-17 pose, more accurate | <https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip> | `f7fbb6c5…0b741` | `5c0a4bf6…3e3c` | Not stated separately (repository: Apache-2.0) | body7 |

Full hashes are in `models.toml`. Citations: YOLOX (Ge et al., 2021, arXiv:2107.08430); RTMPose (Jiang et al., 2023, arXiv:2303.07399); Human-Art (Ju et al., CVPR 2023).

## How weights are fetched

`models.toml` lists every model with its source URL and SHA-256 values. `fim models fetch` downloads a model from that URL, verifies the download and the extracted ONNX file, and stores it as `models/<name>.onnx` (git-ignored). The image build does this in its own stage, so that source changes do not download the weights again, for the models named in the `FIM_MODELS_FETCH` build argument (default `yolox_m_humanart,rtmpose_s,yolox_s`); nothing is downloaded at runtime. `fim models verify` re-checks the files.

## Defaults

- **Detector:** `yolox_m_humanart`. On CAUCAFall it found 77 % of people labelled as fallen versus 63 % for `yolox_s` (see [`docs/benchmarks/perception.md`](docs/benchmarks/perception.md)). Its training data (Human-Art) is non-commercial: installations with commercial use should set `FIM_PERSON_DETECTOR_MODEL=yolox_s`.
- **Pose:** `rtmpose_s` (body7). `rtmpose_m` is available for more accuracy at higher CPU cost.
