# Evaluation

`fim eval` measures the fall rules on public datasets, with the current settings and
models: event-level recall, false alerts per hour and alert latency. The results are
measurements on stated datasets, never clinical accuracy, and the public clips are not a
home: staged falls by young adults, short clips, one camera position per scene.

## Running it

Download the datasets yourself (sources and terms in [`DATASETS.md`](DATASETS.md)), then:

```bash
docker compose --profile dev run --rm dev uv run --frozen \
  fim eval --dataset gmdcsa24 --root <GMDCSA24 folder>
```

`--dataset` is `gmdcsa24`, `urfd` (UR Fall, cam0 videos and frame labels) or `caucafall`.
`--limit N` evaluates an evenly spaced subset. Each video is analysed with `fim analyze`;
the report goes to `artifacts/eval/<dataset>/report.{json,md}`.

## Method

- **Ground truth:** the datasets' own labels, as time ranges per video.
  - GMDCSA24: the fall class ranges of `Fall.csv` (for example `Falling (SW)[3.4 to 6]`).
  - UR Fall: from the first "falling" frame to the last "falling" or "lying" frame of each
    fall sequence. Lying down on purpose in the ADL sequences is not a fall.
  - CAUCAFall: the frames labelled "fall" (from the fall's onset to the end of the clip).
- **Alerts:** `possible_fall` and `person_down` events that would be notified (near-misses
  are not). Head-impact events are not scored: no dataset labels head contacts.
- **Matching:** an alert detects a fall when it starts within 2 s before the fall starts
  to 2 s after it ends (`--before-s`, `--after-s`). Each fall is matched once; further
  alerts for the same fall are duplicates (the service would aggregate them into one
  message). Any other alert is a false alert.
- **Latency:** from the start of the labelled fall to the frame at which the alert was
  raised, in media time. It includes the rule's confirmation (lying for 0.8 s) but not
  delivery.
- **False alerts per hour:** false alerts divided by the total duration of all clips, falls
  included. The clips are short, so this extrapolates from a few minutes of video.
- **Clip ends:** analysis stops with the file. A fall in the last second of a clip can be
  missed because the rule needs the frames after it; in the service, frames keep coming.

## Results

Run of 2026-11-10: default settings and models (`yolox_m_humanart`, `rtmpose_s`, alert
threshold 0.5), no zones, all videos of each dataset.

| Dataset | Videos | Falls detected (recall) | False alerts | Latency median / p95 / max |
|---|---|---|---|---|
| GMDCSA24 | 160 (21.5 min) | 65/79 (82 %) | 5 (14 per hour) | 1.2 / 2.3 / 2.8 s |
| UR Fall (cam0) | 70 (6.6 min) | 18/30 (60 %) | 1 (9 per hour) | 1.2 / 1.9 / 2.1 s |
| CAUCAFall | 100 (16.6 min) | 22/50 (44 %) | 3 (11 per hour) | 1.2 / 2.9 / 4.0 s |

Reading the numbers:

- **Falls towards the camera are the weak spot.** CAUCAFall by kind of fall: forward
  0/10, sitting 3/10, backwards 6/10, left 6/10, right 7/10. A person falling towards or
  away from the camera keeps an upright, narrow box in 2D for the whole fall.
- **Once raised, alerts are fast:** about 1.2 s after the fall starts (median), within
  4 s in every case. Delivery adds the outbox and Telegram time.
- **False alerts come from lying down quickly** (GMDCSA24: sitting on a bed and lying down
  within a second). On site a safe zone over the bed turns these into near-misses; these
  runs use no zones.
- **The hourly rates extrapolate from 6 to 22 minutes of clips** of staged activities.
  The home calibration period (see [`pilot.md`](pilot.md)) is the real measure.
- One GMDCSA24 row (`Subject 4/Fall/15.mp4`) has no time range for its fall; it is taken
  from the end of the previous activity to the end of the clip.
- These match the per-video counts of [`benchmarks/falls.md`](benchmarks/falls.md)
  (GMDCSA24 65/79, CAUCAFall 22/50). UR Fall is lower there (14/30); the cause of the
  difference was not investigated.
