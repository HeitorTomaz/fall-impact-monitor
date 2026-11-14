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

## 2026-11-14: four cameras on four CPUs (#68)

Four cameras analysed together in one process, as the service does (one pipeline per
camera, each in its own thread), in a container limited to `--cpus=4 --memory=8g` on the
desktop below. Each camera reads a video at media time and analyses it as fast as it can;
the **real-time factor** is video seconds handled per wall-clock second (at least 1.0
keeps up). Videos: 60 s per camera, Multiple Cameras Fall Dataset views re-encoded at their
real 30 fps (720×480): four scenes with a person for part of the time, and an empty room
whose furniture yields a weak person detection (score 0.11 to 0.26) now and then.

**ONNX Runtime threads.** By default every session starts one thread per core and spins
while it waits. With eight sessions (detector and pose of four cameras) on four CPUs,
four cameras with people reached a real-time factor of 0.22 over 15 s; with one thread per
session, 0.86 to 1.18. `FIM_INFERENCE_THREADS` (default: the CPUs divided by the cameras)
now sets this.

With one thread per session (lowest camera; the others were close):

| Analysed fps with a person | Person detector | 4 cameras with people | 1 with a person, 3 empty | 4 empty |
|---|---|---|---|---|
| 8 | every analysed frame | 0.55 | 0.48 | 0.48 |
| 8 | at most 4 per second (default) | 0.72 | 0.49 | 0.48 |
| 6 | every analysed frame | 0.68 | 0.66 | 0.65 |
| 5 | every analysed frame | 0.79 | 0.72 | 0.72 |
| 4 | every analysed frame | 0.98 | 0.94 | 0.95 |

Reading the numbers:

- **Four CPUs hold about 14 to 15 person detections per second in total** (about 0.27 s
  of one core each, with `yolox_m_humanart` on one thread); pose is small beside it.
- **The empty room costs as much as a room with people**: its weak detection keeps the
  camera at the person rate (7.5 analyses per second instead of 3). Any detection, however
  weak, counts as a person for the analysis rate. Counting only detections that can start
  a track (`FIM_TRACKER_NEW_TRACK_MIN_SCORE`) would let empty rooms drop to 3 per second,
  but on the public datasets 16 of 246 labelled falls had seconds with only weak
  detections (9 of 48 on the cameras mounted high), so it is not adopted without a
  decision (recall first).
- **At the defaults, four cameras do not keep up on four CPUs**: each analyses the latest
  frame as fast as it can, about 4 to 6 per second instead of 8. That is above the minimum
  rate, so monitoring is not degraded, but the dataset sweep shows recall falling at 6, 5
  and 4 analyses per second (`docs/evaluation.md`). A faster CPU, the empty-room change
  above or a lighter detector are the ways to keep 8.
- The rates the sweep compared (detector-rate sweep, 2026-11-13 to 14): at 8 analyses per
  second, capping the detector at 4 per second kept recall (UR Fall 24/30 both,
  CAUCAFall 40 vs 39/50, GMDCSA24 67 vs 70/79) and added false alerts (UR Fall 5 vs 2,
  GMDCSA24 12 vs 11, MCFD 24 vs 20), against about a third more capacity with people in
  view. Lower analysis rates lose recall (UR Fall 22, 21 and 17/30 at 6, 5 and 4). The
  defaults stay at 8 analyses and at most 4 detections per second.

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
