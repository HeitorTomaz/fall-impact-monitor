# Pipeline benchmark

What one camera costs, measured with `fim benchmark pipeline`: decoding, then detection,
tracking and pose on every frame, resized to `FIM_CAPTURE_MAX_ANALYSIS_WIDTH_PX` as the
service does. Public videos only; reproduce with the command below on your own server
before adding cameras.

```bash
docker compose run --rm -v /path/to/videos:/media:ro app \
  fim benchmark pipeline --source /media/<video>.mp4 --frames 200
```

## Desktop x86-64, 4 cores available to Docker

Docker Desktop (WSL 2), ONNX Runtime 1.31 CPU, default models (`yolox_m_humanart`,
`rtmpose_s`). Video: GMDCSA24 Subject 1, Fall 01 (1280×720, 29.6 fps, one person).
Date: 2026-11-10.

| Measure | Value |
|---|---|
| Decoding | 2.9 ms per frame (p95 3.6) |
| Detection + tracking + pose | 124 ms per analysed frame (p95 164) |
| Of which person detection | about 109 ms; pose about 6 ms per person |
| Highest analysis rate | 7.9 frames per second |
| Full rate (every frame) keeps up with 29.6 fps | no |
| Cameras analysed at 5 fps with people (estimate) | 1.4 |
| CPU | about 4 cores busy while analysing |
| Peak memory | 660 MB |

Reading the numbers:

- **The person detector is the cost.** Decoding and pose are small.
- **One camera at the usual 5 analyses per second fits**, using about 70 % of the four
  cores (30 decodes and 5 analyses per second). A second camera does not fit at that rate.
- **Full-rate analysis cannot keep up** with a 30 fps stream on this class of machine;
  analysis falling behind real time is issue #38.
- **Other load matters:** the same run with a second copy of the stack analysing a
  simulated camera on the same machine measured 367 ms per analysed frame and
  2.7 frames per second.
