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
`sfu_public` / `sfu_public_clear` (real falls of older adults, locally annotated; see
`DATASETS.md`) or `mcfd` (Multiple Cameras Fall Dataset: eight cameras mounted high; its files declare
120 fps, so players show them four times too fast; the authors' paper gives the capture
rate, 30 fps, so its media time is scaled by the declared rate divided by 30, through
`fim analyze --media-time-scale`).
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
    fall sequence (fall-06 from 0.9 s, see below). Lying down on purpose in the ADL
    sequences is not a fall. Its videos
    hold the depth map on the left and the colour image on the right: only the right
    half is analysed (`fim analyze --crop-px`, set by `fim eval`; since 2026-11-16, #93).
  - CAUCAFall: from 1.0 s before the first frame labelled "fall" (not before the clip's
    start) to the last one (usually the end of the clip; FallForwardS9 to the clip's end,
    its labels stop while the man still lies on the floor). The dataset labels a frame
    "fall" once the person is down, so the falling motion comes before it (#94,
    `CAUCAFALL_ONSET_LEAD_S`). Frame times use the file's own frame rate, never an
    assumed one.
- **Alerts:** `possible_fall` and `person_down` events that would be notified (near-misses
  are not). Head-impact events are not scored: no dataset labels head contacts.
- **Matching (since 2026-11-15, #81):** an alert shows the instant at which the fall was
  identified (its event start). It detects a fall when that instant lies from 2 s before
  to 5 s after the labelled start of the fall (`--before-s`, `--after-start-s`) **and** it
  was sent at most 30 s after that start (`--deadline-s`): an alert may come late without
  harm (the maintainer's budget is 30 s, 60 s at most), what matters is many correct
  alerts and few false ones. The report also gives the falls detected with a 60 s deadline
  (`--late-deadline-s`). Each fall takes one alert of its own (the one sent first); other
  alerts whose instant lies in a fall's labelled range (2 s margins, `--after-s` after its
  end) are duplicates (the service would aggregate them into one message). Any other
  alert is a false alert. Before 2026-11-15, an alert detected a fall when its event
  started within 2 s before the fall's start to 2 s after its labelled end (which often
  includes the lying period), with no deadline: older sections use that criterion.
- **Latency:** from the start of the labelled fall to the frame at which the alert was
  sent, in media time. It includes waiting for later frames (the rule's confirmation, the
  full-rate re-analysis) but not processing delays (offline analysis is not bound to real
  time) nor delivery. An alert sent before the labelled start counts as negative latency.
- **False alerts per hour:** false alerts divided by the total duration of all clips, falls
  included. The clips are short, so this extrapolates from a few minutes of video.
- **Playback speed:** rules measure speeds and durations in media time, so a file that
  plays faster or slower than it was filmed shifts every threshold. Checked on
  2026-11-14 against the authors' statements, the listed lengths and the content:
  - MCFD declares 120 fps but was captured at 30 (the authors' paper): scaled by 4.
  - GMDCSA24 (30 fps, some videos 15): the lengths in its CSVs match the files.
  - UR Fall (Kinect, 30 fps): the files declare 30.
  - CAUCAFall declares 20 fps while its paper says the camera captures 23; its frame
    count matches the paper's labelled frames, so the files may play up to 13 % slow.
    Not confirmed by the content (pose-based step counts are too noisy at this
    resolution); not corrected.
- **Clip ends:** the clips end a few seconds after the fall (0 to 3.5 s on the SFU public
  videos), while processing waits after it (the rule's lying confirmation, the full-rate
  re-analysis, the person-on-the-floor timers). Since 2026-11-15 `fim eval` analyses each
  clip followed by its last frame repeated for the deadline (`--pad-end-s`, default the
  deadline; `fim analyze --pad-end-s`), timestamps continuing at the frame interval and
  every padded line flagged in `tracks.jsonl`. This is evaluation only: the service never
  invents frames. A frozen frame looks like a perfectly still person, so the report counts
  apart the detections and false alerts raised on padding. The full-rate re-analysis
  decodes only real frames. False alerts per hour still divide by the clips' own length.

## Results

### 2026-11-16: MCFD ground truth checked against the videos (#99)

Every MCFD fall was checked frame by frame on the eight cameras (1 s before the labelled
start, the start, +0.5 s, +1 s and the labelled end, cameras aligned by the report's
delays), and every table of the technical report and of the erratum was compared with
`evaluation/mcfd.py`. Two errors of the report, not covered by the erratum, were fixed:

- **Scenario 23 has two falls** (report p. 21: falling 1520–1595 and 3574–3614, each
  followed by getting up); only the second was counted.
- **Scenario 4's periods are in camera 4's frames** (delay 0), not camera 6's as the
  report says: with camera 6 every label came 2.7 s before the motion.

Scenarios 6 and 13 start their falling period about 1 s before the visible descent (inside
the scoring window; left as labelled). Replayed on the #77 analyses (cameras 1 and 5,
30 s padding): falls detected 22/48 → 23/50, false alerts 25 → 24, median latency 1.2 →
1.0 s. Scenario 4's two falls are still missed on both cameras.

### 2026-11-16: UR Fall analysed on its colour half only (#93)

Each UR Fall video is 640×240: the depth map (left) and the colour image (right) side by
side. Until now the whole frame was analysed, so the person appeared twice, each at half
size, once as a grey depth picture. Every UR Fall result before this date, and the UR Fall
windows of the temporal classifier, came from that double picture. The full UR Fall run
was made again on the colour half (defaults of #77, camera height 0.8 m, #78 profiles):

| UR Fall | Falls detected | False alerts (on padding) | Latency median / p95 |
|---|---|---|---|
| Whole frame (#78 replay) | 28/30 | 4 (1) | 0.9 / 1.7 s |
| Colour half | 28/30 | 9 (4) | 0.9 / 1.9 s |

- **Gained:** fall 27. **Lost:** fall 17: the person falls towards the camera and ends
  with only the head and shoulders at the bottom edge of the colour image (box 73×52 px
  touching the edge), so the lying shape is not seen.
- **New false alerts:** people getting down to the floor on purpose, on hands and knees
  or lying down (ADL 31, 32, 38, 39), and a person lying on a sofa (ADL 11); four of the
  nine were raised on the repeated last frame. The colour image shows these postures
  larger and sharper than the half-size double picture did.
- **Labels checked by eye** on all 70 videos (colour half; falls every 0.1 s around the
  labelled start, daily activities 10 frames each and every 0.2 s around each move to the
  floor): every fall has 30 "falling" frames before the "lying" ones, and in fall-06 they
  mostly cover the man still sitting on the chair's edge; its fall now starts at 0.9 s
  (`URFD_FALL_START_S`), with no change to the scores. No unlabelled fall; the deliberate
  moves to the floor in daily activities (ADL 30–35, 38–40) are slow and controlled.

### 2026-11-16: CAUCAFall falls start before the first "fall" frame (#94)

Every CAUCAFall video was checked by eye against its labels (50 falls: person crops every
0.1 s from 2.5 s before the first "fall" frame; 50 daily activities: 10 frames each plus
every frame whose box is wider than tall). The first "fall" frame shows the person already
down: the first visible downward movement comes a median **1.0 s** earlier (quartiles
0.7 s and 1.2 s, from 0.1 s to 1.6 s; by fall type 0.95 to 1.2 s). The truth now starts
1.0 s earlier. A first estimate from the dataset's per-frame boxes (box height below 85 %
of standing, 0.85 s) measured a later point of the descent. Also:

- **FallForwardS9:** the labels go back to "nofall" at 5.15 s while the man lies face
  down without moving to the clip's end (7.40 s); the fall now lasts to the clip's end.
- **FallSittingS7:** her first "fall" frame already shows her tipping over the chair; the
  1.0 s lead would start the fall while she still stretches on it. Her fall starts at
  2.25 s, where it is visible (`CAUCAFALL_FALL_START_S`).
- No daily-activity frame is labelled "fall" and none shows a fall; no fall video holds a
  second, unlabelled fall.
- Label files outnumber the decoded frames by one in 97 videos (by 23 in
  FallBackwardsS6), so some falls end up to 1.15 s after the clip: no effect on scoring.
- None of the videos is interlaced (adjacent rows differ less than rows two apart, ratio
  0.54 to 0.62).

Replayed on the #77 analyses (each dataset's camera height, 30 s padding):

| CAUCAFall | Falls detected | False alerts | Latency median / p95 |
|---|---|---|---|
| Fall from the first "fall" frame | 45/50 | 4 | 0.25 / 2.3 s |
| Fall 1.0 s earlier, FallForwardS9 to its end | 45/50 | 4 | 1.25 / 3.3 s |

The same falls are detected; latency now counts from the start of the motion. The change
matters more for the temporal classifier (#15): its training windows holding the falling
motion but ending before the first "fall" frame were labelled "no fall".

### 2026-11-15: camera height in the fall rules (#78)

`fim eval` analyses each dataset at its camera height (`CAMERA_HEIGHT_M` in
`cli/evaluate.py`; only CAUCAFall publishes it, 2.15 m; the others were estimated from the
videos: UR Fall and GMDCSA24 about 0.8 m and level, MCFD about 2.4 m and SFU about 2.8 m,
both tilted down). Measured by replay on the #77 analyses.

A rule that counted a shortened trunk as lying under cameras of 1.8 m or more (a fall away
from a high camera keeps the trunk upright in the image and barely moves the hips down
it) passed on synthetic pinhole tracks but not on the videos:

| | MCFD | CAUCAFall | UR Fall | GMDCSA24 | SFU |
|---|---|---|---|---|---|
| Without the rule | 19/48 · 25 false | 42/50 · 5 | 27/30 · 5 | 72/79 · 26 | 7/39 · 3 |
| With it, each dataset's height | 20 · 32 | 41 · 6 | 27 · 5 | 72 · 26 | 7 · 3 |
| With it, 2.5 m everywhere | 20 · 32 | 41 · 6 | 28 · **13** | 73 · 26 | 7 · 3 |

Tuned on the training scenes only (trunk at most 0.6 of its upright length, box at most
0.85), the held-out scenes gave MCFD 4 → 5 of 10 with 9 → 11 false alerts and CAUCAFall
10 → 9 of 10. Wide-angle lenses (legs spread on the floor keep the box tall), distance
and slow falls (the upright reference shrinks with them) vary too much for one threshold.
**The rule was left out**; the camera height is kept for the learned final score (#79).

**Height profiles (adopted by the maintainer, 2026-11-16).** Eleven settings of the fall
rule were searched by replay (53 and 57 combinations) on the train and validation splits,
one profile for eye-level cameras (UR Fall, GMDCSA24) and one for cameras mounted high
(CAUCAFall, MCFD), keeping the most falls without more false alerts. The search kept:
eye level, hip drop speed 0.6 → 0.8 heights per second, hip drop 0.35 → 0.25 heights, box
aspect for lying 1.25 → 1.5; high, hip drop 0.35 → 0.45 heights, shoulder drop 0.6 → 0.4
and shoulder speed 1.0 → 1.5 (`config_for_height`, interpolated between 1.0 and 2.2 m).

| Split | Eye level (UR Fall + GMDCSA24) | High (CAUCAFall + MCFD; SFU on test only) |
|---|---|---|
| Train + validation, before → after | 83 → 84 / 87 falls, 22 → 14 false | 47 → 53 / 78, 19 → 18 |
| Held out, before → after | 16 → 16 / 22, 9 → 9 | 21 → 21 / 59, 14 → 14 |

On the held-out splits the profiles changed nothing (no gain, no loss; they are small:
22 and 59 falls). Over every video (train and test together, so optimistic): UR Fall 27 →
28 / 30 (false 5 → 4), CAUCAFall 42 → 45 / 50 (5 → 4), GMDCSA24 72 / 79 (26 → 19), MCFD
19 → 22 / 48 (25), SFU 7 / 39 (3).
Kept: under a high camera, the person-on-the-floor rule also counts a person lying head
first towards it; the clips are too short for that rule's 30 to 60 s, so the datasets do
not measure it. An eye-level camera left at the 2.5 m default would raise false alerts
(UR Fall 5 → 13 with the rule above): set the height for such cameras.

### 2026-11-15: weak detections continue tracks, the person rate needs a strong one (#77)

Every dataset at the defaults, before and after #77 (after: `FIM_ANALYSIS_PERSON_HOLD_S`
10 s, `FIM_ANALYSIS_EMPTY_SCENE_FPS` 2, `FIM_FALL_CONFIRMATION_MAX_FPS` 8), with the
matching criterion of #81 (instant from 2 s before to 5 s after the fall's start, sent
within 30 s; clips padded by 30 s). The person detector's output was reused from the #68
sweep; every other stage ran.

| Dataset | Falls detected before → after | Within 60 s, after | False alerts before → after |
|---|---|---|---|
| UR Fall | 25 → 27 / 30 | 27 | 7 → 5 |
| CAUCAFall | 40 → 42 / 50 | 42 | 6 → 5 |
| GMDCSA24 | 67 → 72 / 79 | 72 | 21 → 26 |
| MCFD (cameras 1 and 5) | 15 → 19 / 48 | 19 | 25 → 25 |
| SFU public videos | 6 → 7 / 39 | 8 | 0 → 3 |

- **18 falls gained, 4 lost.** Most gains are falls during which the person was seen only
  through weak detections for a while: before, they had no track there, so the fall rule
  did not see them.
- **Lost through the full-rate re-analysis (2):** an MCFD fall (scene 14) and a CAUCAFall
  fall (sitting, subject 2) were confirmed by the re-analysis before and are not after.
  Both files are MPEG-4 Part 2, whose re-analysis windows do not decode like the file
  (#85): those results change from run to run. The CAUCAFall one also changed with the
  re-analysis rate (12 to 8 per second).
- **Lost (MCFD, a scene with two falls):** the person stays tracked between the falls, an
  alert in between starts the rule's 5 s cooldown for that person, and the cooldown covers
  the start of the second fall. Before, the person's track was lost during the second fall
  and the track-lost rule alerted.
- **Lost (SFU):** the alert was sent about 1 s after the fall, but the instant it reports
  is the start of a suspicion that began 3.6 s before the fall (the person was followed
  throughout), outside the 2 s window; the person-on-the-floor alert follows 31 s after
  the fall (detected within 60 s).
- **Person-on-the-floor rule without weak untracked boxes:** replaying the same analyses
  with the rule counting them again gave identical results on every dataset (measured at a
  3 s hold).

### 2026-11-15: reported instant, sending deadline and clip-end padding (#81)

Every dataset again at the defaults (8 analysed frames per second with a person, person
detector at most 4 per second, rules only), with the new matching criterion and each clip
followed by its last frame repeated for 30 s. The person detector's output was reused from
the #68 sweep (deterministic, same settings); every other stage ran again.

| Dataset | Falls detected (sent within 30 s) | Within 60 s | False alerts | Raised on padding (detections / false) | Latency median |
|---|---|---|---|---|---|
| UR Fall | 25/30 | 25/30 | 7 (63 per hour) | 1 / 2 | 1.0 s |
| CAUCAFall | 40/50 | 40/50 | 6 (22 per hour) | 1 / 0 | 0.7 s |
| GMDCSA24 | 67/79 | 69/79 | 21 (59 per hour) | 0 / 9 | 1.0 s |
| MCFD (cameras 1 and 5) | 15/48 | 15/48 | 25 (41 per hour) | 0 / 0 | 0.9 s |
| SFU public videos | 6/39 | 6/39 | 0 | 1 / 0 | 1.9 s |

- **More false alerts than with the previous criterion** (UR Fall 5 to 7, GMDCSA24 12 to
  21): alerts whose instant lies after the fall's start plus 5 s are no longer counted as
  detections, and the ones in no fall's labelled range are false; the repeated last frame
  adds some (a frozen image of a person lying still raises person-on-the-floor alerts).
- **Few falls need more than 30 s** (GMDCSA24 2): almost every alert is sent within a few
  seconds.
- Padding recovered a detection on UR Fall, CAUCAFall and SFU each (processing that waited
  after the clip's end).

### 2026-11-14: real falls of older adults (SFU public videos, #72)

The 39 views with a visible fall (28 `clear`, 7 `uncertain`, 4 `assumed`; see
[`DATASETS.md`](DATASETS.md)), analysed during the detector-rate sweep (#68) at 4 to 8
analysed frames per second with a person, with and without a detector cap. Falls only, so
this measures recall, not false alerts.

**Playback speed checked:** the views are 30 fps files from cameras of different rates.
About a quarter repeat each image three or four times (cameras of roughly 8 to 10 fps
re-encoded with their timing kept); the others hold 23 to 30 distinct images per second.
Both kinds play in real time: a staff member crosses one of the repeating views at
about 0.75 of their height per second (an ordinary walk), and views of the same fall
from cameras of different rates show it over the same time span.

| Analysed fps with a person | Falls detected (all visible) | `clear` only |
|---|---|---|
| 8 (detector every frame) | 5/39 | 5/28 |
| 8 (detector at most 4 per second) | 6/39 | 6/28 |
| 6 | 5/39 | 5/28 |
| 5 | 6/39 | 6/28 |
| 4 | 4/39 | 4/28 |

**The rules miss most real falls of older adults in these clips.** In the missed views
they raise nothing at all, not even a near-miss. Looking at the tracks around each
missed fall:

- **The clips end 0 to 3.5 s after the fall** (median 2 s). In the service frames keep
  coming, so a fall that the rules need a few more seconds for is cut off here. This
  explains part of the gap, not all of it.
- **Slow falls that end sitting or slumped**: sliding out of a wheelchair, a transfer
  that goes wrong, a slow descent with a walker. Where the person is tracked through the
  fall, the hip drop is mostly under 0.3 person heights and the torso stays below 45°
  from vertical at the end: neither a fast drop nor lying.
- **The tracker loses the person during the fall** in about a quarter of the missed views (a track
  before the fall, another after it, sometimes several people close together with staff
  around), and in about a sixth no pose is found at all: far away, small, low resolution
  (some views are 352 × 240), blurred or backlit.

This is the most important limitation found so far for the project's purpose. Follow-ups:
falls that end sitting, clip ends in the evaluation (a separate "candidate" recall), and
keeping the person through a fall; the temporal classifiers (#15) are compared on these
views too.

### 2026-11-14: cameras mounted high (MCFD, #69)

`fim eval --dataset mcfd --limit 48`: cameras 1 and 5 of all 24 scenes (one fall each),
analysed during the detector-rate sweep (#68) with media time scaled to the real 30 fps.
Alert threshold 0.5, no zones.

| Analysed fps with a person | Falls detected (recall) | Camera 1 | Camera 5 | False alerts | Latency median / p95 |
|---|---|---|---|---|---|
| 8 (detector every frame) | 15/48 (31 %) | 6/24 | 11/24 | 20 (33 per hour) | 0.9 / 7.9 s |
| 8 (detector at most 4 per second) | 15/48 (31 %) | 6/24 | 9/24 | 24 (40 per hour) | 0.9 / 6.3 s |
| 6 (detector every frame) | 14/48 (29 %) | 5/24 | 10/24 | 19 (31 per hour) | 1.3 / 6.8 s |

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
