# Pipeline benchmark

What one camera costs, measured with `fim benchmark pipeline`: decoding every frame, then
detection, tracking and pose at the person-visible analysis rate in video time, resized to
`FIM_CAPTURE_MAX_ANALYSIS_WIDTH_PX` as the service does. The command runs twice: with the
person detector on every analysed frame, then at `FIM_PERSON_DETECTOR_MAX_FPS` (#68).
Before 2026-11-13 it analysed every frame with the detector on each. Public videos only;
reproduce with the command below on your own server before adding cameras.

```bash
docker compose run --rm -v /path/to/videos:/media:ro app \
  fim benchmark pipeline --source /media/<video>.mp4 --frames 200
```

## 2026-11-13: person detector at its own rate (#68)

Same desktop and video as below, 120 analysed frames at 8 per second of video:

| Person detector | Detection + tracking + pose per analysed frame | Cameras at 8 fps with people (estimate) |
|---|---|---|
| Every analysed frame | 107 ms (p95 130) | 1.1 |
| At most 4 per second (default) | 68 ms (p95 128) | 1.6 |

- Over four GMDCSA24 videos (two falls, two daily activities), 46 of 102 analysed frames
  were followed by pose alone; on the others the detector was due (45) or ran at once
  because a box changed a lot (10). At 8 analyses and 4 detector runs per second, at most
  half the frames can be followed.
- The p95 stays close: a frame on which following fails pays for the pose attempt and
  then the detector.
- The peak memory the command reports (about 840 MB) includes both runs' models in one
  process; the service loads them once.

## Desktop x86-64, 4 cores available to Docker

Docker Desktop (WSL 2), ONNX Runtime 1.31 CPU, default models (`yolox_m_humanart`,
`rtmpose_s`). Video: GMDCSA24 Subject 1, Fall 01 (1280×720, 29.6 fps, one person).
Date: 2026-11-10.

| Measure | Value |
|---|---|
| Decoding | 2.9 ms per frame (p95 3.6), about 0.09 s per second of video (decoding and conversion, measured as elapsed time) |
| Detection + tracking + pose | 124 ms per analysed frame (p95 164) |
| Of which person detection | about 109 ms; pose about 6 ms per person |
| Highest analysis rate | 7.9 frames per second |
| Full rate (every frame) keeps up with 29.6 fps | no |
| Cameras analysed at 5 fps with people (estimate) | 1.4 |
| Cameras analysed at 8 fps with people (estimate, computed from the same run) | 0.93 |
| CPU | about 4 cores busy while analysing |
| Peak memory | 660 MB |

Reading the numbers:

- **The person detector is the cost.** Decoding and pose are small.
- **One camera at 5 analyses per second fits**, using about 70 % of the four cores
  (30 decodes and 5 analyses per second). A second camera does not fit at that rate.
- **At the default of 8 analyses per second with a person in view** (since #54), one
  camera needs about 1.08 s of these four cores every second: this machine reaches about
  7.4 analyses per second. The rate is a target: a machine that cannot reach it analyses
  as fast as it can, always the latest frame, and reports degraded monitoring only below
  `FIM_ANALYSIS_MIN_FPS`. A slower server reaches fewer.
- **Full-rate analysis cannot keep up** with a 30 fps stream on this class of machine;
  analysis falling behind real time is issue #38.
- **Other load matters:** the same run with a second copy of the stack analysing a
  simulated camera on the same machine measured 367 ms per analysed frame and
  2.7 frames per second.
