# Capture timing protocol

How to characterize the timing of a camera stream before trusting it for kinematics. Results that come from a private installation stay in the git-ignored `local/` folder; this page only describes the method and anonymized conclusions.

## Why

Detection uses short time windows (a fall lasts about one second; a head impact is shorter than a frame). Consumer cameras may deliver frames in bursts, repeat timestamps, or jump forward, and different tools hide or expose this differently. A recording that "looks smoother" can simply have discarded the timing (AVI files store no per-frame timestamps). Every measurement below therefore records three time domains separately: source PTS, monotonic arrival time, and UTC wall clock.

## Tool

`fim probe` reads a file or stream with PyAV, classifies every frame timestamp (`ok`, `zero_delta`, `backward`, `outlier_gap`, `missing`) and writes `frames.csv`, `summary.json` and `summary.md` under `artifacts/probe/`.

```bash
docker compose --profile dev run --rm dev uv run --frozen fim probe \
  --source rtsp://mediamtx:8554/cam1sub --duration-s 120 --out-dir artifacts/probe/<name>
```

An `outlier_gap` is a positive delta larger than 4× the mean of the last 50 non-negative deltas. The mean of every delta (including zero deltas and gaps) is used because bursty sources hold timestamps and then catch up; a median would drift below the real frame interval.

## Experiments

Run each for at least 120 s, in daylight and in low light, at every frame rate the camera offers.

| # | Comparison | Command variation | Question answered |
|---|---|---|---|
| 1 | Direct vs relay | `--source rtsp://<camera>/…` vs `rtsp://mediamtx:8554/<path>` | Does MediaMTX change timestamps? |
| 2 | Main stream vs substream | `cam1` vs `cam1sub` | Which stream has better timing for analysis? |
| 3 | Video-only vs all media | `--video-only` (default) vs `--no-video-only` | Does RTCP-based audio/video sync step the PTS? |
| 4 | Source PTS vs wall clock | default vs `--wallclock` | How much of the irregularity is arrival jitter? |
| 5 | Ground truth | film a running millisecond stopwatch or a blinking LED; compare frame contents against PTS | Do PTS reflect capture time? |
| 6 | Recorder robustness | leave MediaMTX recording for 24–48 h; check its log for recorder errors and segment gaps | Is the MediaMTX recorder safe for this camera? |

Also search old FFmpeg logs for "Non-monotonic DTS" messages: they indicate backward timestamps that a muxer clamped.

## Simulated camera

Without a camera, the `sim` Compose profile loops a local file into MediaMTX (`rtsp://mediamtx:8554/cam1`). The publisher re-timestamps packets, so the simulated stream does not reproduce a camera's timing; it only exercises the relay, the recorder and clip export.

## Findings so far (anonymized)

- Preliminary local captures of the reference camera showed irregular source timestamps: runs of identical timestamps followed by forward jumps, roughly once per second. Decoding showed no H.264 errors, so frames were not lost after encoding.
- A recording made with a tool that writes AVI looked smoother only because AVI stores a fixed frame rate.
- The MediaMTX recorder recorded a stream with these timestamp patterns without errors in a short simulated run; the 24–48 h run is still pending.
- Analysis therefore never uses a zero, negative or outlier delta for kinematics; arrival time is recorded for every frame so the two domains can be compared.

## Open

The ground-truth test (experiment 5) decides which time domain the confirmation stage uses for kinematics.
