# Evaluation

`fim eval` measures the fall rules on public datasets, with the current settings and
models: event-level recall, false alerts per hour and alert latency. The results are
measurements on stated datasets, never clinical accuracy, and the public clips are not a
home: staged falls by young adults, short clips, one camera position per scene.

## Running it

Download the datasets yourself (sources and terms in [`DATASETS.md`](DATASETS.md); `fim datasets fetch` does it for GMDCSA24 and UR Fall), then:

```bash
docker compose --profile dev run --rm dev uv run --frozen \
  fim eval --dataset gmdcsa24 --root <GMDCSA24 folder>
```

`--dataset` is `gmdcsa24`, `urfd` (UR Fall, cam0 videos and frame labels), `caucafall` or
`mcfd` (Multiple Cameras Fall Dataset: eight cameras mounted high; its files declare
120 fps but were captured at about 30, so its media time is scaled by the declared rate
divided by 30, through `fim analyze --media-time-scale`).
`--limit N` evaluates an evenly spaced subset. Each video is analysed with `fim analyze`
(with the settings of the environment it runs in) into its own folder; the report goes to
`artifacts/eval/<dataset>/report.{json,md}`. `--reuse` scores the saved analyses again
(for a different matching window, for example); a video is analysed again when its saved
analysis is incomplete, belongs to another file or was made with another perception
configuration (recorded in each video's `meta.json`).

### Replay: rule changes in minutes

`--replay` runs the current detectors again on the saved analyses of `--out-dir`, without
perception, and writes `report.replay.{json,md}` (each video's events go to
`videos/<video>/replay/events.jsonl`). Use it for changes to the rules and their
settings (thresholds, timings, which detectors run); a full run takes hours, a replay
minutes.

- It is exact: `tracks.jsonl` keeps every frame read (perception results at full
  precision for the analysed ones) and `confirmations.jsonl` the perception of each
  full-rate re-analysis window, and the replay runs the same steps as the analysis.
- It refuses analyses made with another perception configuration (models, analysis
  and detector rates, tracker, re-analysis settings), naming the variables that differ,
  and analyses saved before replay existed. Model or perception changes need a full run.
- A rule change can create fall near-misses whose re-analysis window was never saved.
  They are not analysed again; the report says how many, and a full run measures them.
- Perception results do not depend on the rules offline: `fim analyze` does not raise the
  analysis rate for a moving head as the service does.

## Method

- **Ground truth:** the datasets' own labels, as time ranges per video.
  - GMDCSA24: the fall class ranges of `Fall.csv` (for example `Falling (SW)[3.4 to 6]`;
    one row writes `[a b]`). Falls that touch are one fall. A fall row without any fall
    is an error, because every video in `Fall/` is a fall.
  - UR Fall: from the first "falling" frame to the last "falling" or "lying" frame of each
    fall sequence. Lying down on purpose in the ADL sequences is not a fall.
  - CAUCAFall: from the first to the last frame labelled "fall" (usually to the end of
    the clip). Frame times use the file's own frame rate, never an assumed one.
- **Alerts:** `possible_fall` and `person_down` events that would be notified (near-misses
  are not). Head-impact events are not scored: no dataset labels head contacts.
- **Matching:** an alert detects a fall when it starts within 2 s before the fall starts
  to 2 s after it ends (`--before-s`, `--after-s`). Each fall takes one alert of its own
  (the one raised first); alerts left inside a fall's window are duplicates (the service
  would aggregate them into one message). Any other alert is a false alert.
- **Latency:** from the start of the labelled fall to the frame at which the alert was
  raised, in media time. It includes the rule's confirmation (lying for 0.8 s) but not
  processing delays (offline analysis is not bound to real time) nor delivery. An alert
  raised before the labelled start counts as negative latency.
- **False alerts per hour:** false alerts divided by the total duration of all clips, falls
  included. The clips are short, so this extrapolates from a few minutes of video.
- **Clip ends:** analysis stops with the file. A fall in the last second of a clip can be
  missed because the rule needs the frames after it; in the service, frames keep coming.

## Results

### 2026-11-14: cameras mounted high (MCFD, #69)

`fim eval --dataset mcfd --limit 48`: cameras 1 and 5 of all 24 scenes (one fall each),
analysed during the detector-rate sweep (#68) with media time scaled to the real 30 fps.
Alert threshold 0.5, no zones.

| Analysed fps with a person | Falls detected (recall) | Camera 1 | Camera 5 | False alerts | Latency median / p95 |
|---|---|---|---|---|---|
| 8 (detector every frame) | 15/48 (31 %) | 6/24 | 11/24 | 20 (33 per hour) | 0.9 / 7.9 s |
| 8 (detector at most 4 per second) | 15/48 (31 %) | 6/24 | 9/24 | 24 (40 per hour) | 0.9 / 6.3 s |

- **Recall is far below the other datasets** (76 to 89 %). The cameras are wide-angle
  and mounted high in the corners of the room: the median person box is 110 px high in
  a 720 × 480 frame (interlaced, low quality). A likely cause, not yet checked fall by
  fall: a fall towards or away from a camera seen from above moves the hips little down
  the image.
- **False alerts:** the scenes also hold sitting and lying down on a sofa, crouching and
  other daily activities on purpose, and the dataset does not label their times, so the
  false alerts were not attributed to activities. Safe zones over sofas (on site) turn
  lying down there into near-misses.
- These are the scenes the temporal classifiers (#15) are also compared on.

### 2026-11-12: full run at 8 analysed frames per second with the new rules

`fim eval` on all videos with the default models, the new default analysis rate of
8 frames per second with a person (#54) and the fall rules for cameras mounted high
(#49, below). No zones; alert threshold 0.5.

| Dataset | Falls detected (recall) | False alerts | Latency median / p95 / max |
|---|---|---|---|
| GMDCSA24 | 70/79 (89 %) | 12 (34 per hour) | 1.0 / 2.2 / 4.0 s |
| UR Fall (cam0) | 24/30 (80 %) | 2 (18 per hour) | 1.0 / 1.6 / 1.8 s |
| CAUCAFall | 38/50 (76 %) | 6 (22 per hour) | 0.7 / 2.4 / 4.9 s |

- Over the three datasets: **132/159 falls (83 %), was 105/159 (66 %)** on 2026-11-10;
  false alerts 20, was 9. UR Fall had one duplicate alert for a detected fall.
- After the review fix for stooping with the hips still (a shoulder drop alone now
  needs a person who looks lying or whose hips went down 0.15 heights), the same
  stored tracks replayed give the same recall with 11 / 2 / 5 false alerts.
- The false alerts are mostly kneeling, picking up an object and lying down quickly on a
  bed; on site, safe zones over beds and sofas turn the last ones into near-misses.
- Not included: the identity hand-over by count (#59) and the full-rate re-analysis of
  near-misses (#64), measured separately.

### 2026-11-12: rules for cameras mounted high (#49)

The perception of the 2026-11-10 run below (same videos, detector, tracker and pose,
5 analysed frames per second with a person) replayed through the new fall rules: a signed
torso angle, and the shoulders' drop and speed with higher thresholds than the hip's (see
[`fall-detection.md`](fall-detection.md#possible-fall-possible_fall)). Scoring as above.

| Dataset | Falls detected (recall) | False alerts | Latency median / p95 / max |
|---|---|---|---|
| GMDCSA24 | 68/79 (86 %), was 65 | 8 (22 per hour), was 5 | 1.1 / 2.3 / 2.8 s |
| UR Fall (cam0) | 21/30 (70 %), was 18 | 2 (18 per hour), was 1 | 1.1 / 1.9 / 2.1 s |
| CAUCAFall | 35/50 (70 %), was 22 | 4 (15 per hour), was 3 | 1.0 / 2.6 / 4.0 s |

- **Forward falls:** CAUCAFall forward 9/10 (was 0/10), sitting 4/10, backwards 7/10,
  left 8/10, right 7/10. Most forward falls end head first towards the camera, mounted
  high: the shoulders end below the hips in the image, which the unsigned angle read as
  upright. The signed angle alone gave CAUCAFall 34/50 and UR Fall 19/30 with no added
  false alert.
- **The shoulders' drop** added 6 falls and 5 false alerts over the three datasets
  (kneeling, picking up an object, sitting down quickly). Thresholds of 0.5 to 0.7
  heights were compared; 0.6 heights and 1.0 heights per second were kept, as recall
  comes first. Treating any inverted torso as lying added false alerts and no fall, so it
  was not adopted.
- The default analysis rate is now 8 frames per second with a person (#54); these
  numbers were measured at 5; the full run at 8 is above.

### 2026-11-10

Run of 2026-11-10: default settings and models (`yolox_m_humanart`, `rtmpose_s`, alert
threshold 0.5), no zones, all videos of each dataset.

| Dataset | Videos | Falls detected (recall) | False alerts | Latency median / p95 / max |
|---|---|---|---|---|
| GMDCSA24 | 160 (21.5 min) | 65/79 (82 %) | 5 (14 per hour) | 1.2 / 2.3 / 2.8 s |
| UR Fall (cam0) | 70 (6.6 min) | 18/30 (60 %) | 1 (9 per hour) | 1.1 / 1.9 / 2.1 s |
| CAUCAFall | 100 (16.6 min) | 22/50 (44 %) | 3 (11 per hour) | 1.2 / 2.9 / 4.0 s |

Reading the numbers:

- **Forward falls were the weak spot.** CAUCAFall by kind of fall: forward 0/10, sitting
  3/10, backwards 6/10, left 6/10, right 7/10. Fixed for most forward falls on
  2026-11-12 (above).
- **Once raised, alerts are fast:** about 1.1–1.2 s after the fall starts (median), within
  4 s in every case. Processing delays and delivery come on top.
- **False alerts come from lying down quickly** (GMDCSA24: sitting on a bed and lying down
  within a second). On site a safe zone over the bed turns these into near-misses; these
  runs use no zones.
- **The hourly rates extrapolate from 6 to 22 minutes of clips** of staged activities.
  The home calibration period (see [`pilot.md`](pilot.md)) is the real measure.
- One GMDCSA24 row (`Subject 4/Fall/15.mp4`) has no time range for its fall; it is taken
  from the end of the previous activity to the end of the clip.
- The per-video counts of [`benchmarks/falls.md`](benchmarks/falls.md) give the same
  totals for GMDCSA24 (65/79) and CAUCAFall (22/50), but they count differently (any alert
  anywhere in a fall video), so the videos behind them differ. UR Fall is lower there
  (14/30); the cause of the difference was not investigated.
