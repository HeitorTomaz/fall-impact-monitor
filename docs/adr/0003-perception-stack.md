# ADR-0003: Perception stack: benchmark-selected detector, OC-SORT, RTMPose on ONNX Runtime

- Status: Accepted
- Date: 2026-11-08

## Context

Detection logic needs, per camera, tracked people with body keypoints. Scenes are indoor and room-scale, with several people, partial occlusion, people far from the camera and people lying on the floor. The system must run on a CPU-only host, allow optional acceleration, and stay portable to ARM64. Model choice is driven by detection quality (ADR-0009), within the CPU budget.

## Decision

- **Person detector: selected by benchmark** on the target host and on representative recordings. Candidates: YOLOX-Tiny/S, RTMDet, D-FINE, and YOLO11/YOLO26. Ultralytics models are used only as **exported ONNX files**; the `ultralytics` package never enters the repository or the image. Criteria: recall of distant, lying and unusually posed people, within the CPU budget.
- **Tracker: OC-SORT** from the Roboflow `trackers` package (Apache-2.0), with appearance re-identification **off**. Its update accepts absolute timestamps, which suits irregular frame timing. Keypoints are carried per track; crossings and occlusions mark the identity `uncertain`.
- **Pose: RTMPose**, top-down on per-person crops, using the most accurate available variant. Lying people (wide boxes) get a 90° rotated crop. Pre/post-processing is copied from rtmlib (Apache-2.0) into `_vendor/` instead of depending on the package, which pulls in two OpenCV distributions and downloads default weights.
- **RTMW** (whole-body, with hand keypoints) is a **post-MVP** addition that will run **only in the confirmation stage**, for hand-to-head evidence, because of its CPU cost. In the MVP the confirmation stage uses RTMPose.
- **Runtime: ONNX Runtime CPU** by default (x86_64 and ARM64). OpenVINO is an optional backend on Intel CPUs with AVX2. The CUDA execution provider is supported only for GPUs of compute capability ≥ 7.5; older generations are not supported by current CUDA/cuDNN releases.
- Weights are pinned by SHA-256 in the model manifest (`MODELS.md`) and downloaded at build time (ADR-0008).

## Alternatives considered

- **MediaPipe Pose Landmarker**: clean licence and active, but its model card puts multiple people, distances beyond 4 m, hidden heads and non-upright people out of scope, which is exactly the post-fall situation.
- **MoveNet MultiPose**: designed for near-range scenes, no longer maintained.
- **ViTPose / Sapiens**: accurate but too heavy for real-time CPU inference.
- **One-stage multi-person pose (e.g. RTMO)**: cost does not grow with people, but small people shrink at its input size; may be revisited by benchmark.

## Consequences

- Top-down pose cost grows with the number of people; the scheduler (ADR-0011, ARCHITECTURE) must prioritize and batch crops across cameras.
- The detector choice is a measured, documented result that can change with hardware; it is configuration, not code.
- Some selected weights have non-commercial or unclear training-data terms; they are documented in `MODELS.md` and never redistributed (ADR-0009).

## References

- https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose
- https://github.com/Tau-J/rtmlib
- https://github.com/roboflow/trackers
- https://github.com/Megvii-BaseDetection/YOLOX
- https://onnxruntime.ai/docs/execution-providers/
