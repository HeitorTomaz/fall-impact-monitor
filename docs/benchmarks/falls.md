# Fall detection benchmark

Results on public datasets only. They describe these recordings and these rules; they are
**not** clinical accuracy figures and do not predict performance in a particular home.
Date: 2026-11-09. Rules as described in [`../fall-detection.md`](../fall-detection.md), at
their defaults (alert threshold 0.5).

## Method

- Perception: the default pipeline (YOLOX-m HumanArt detector, OC-SORT tracker,
  RTMPose-s), with the default adaptive cadence (5 analysed frames per second while a
  person is visible), on decoded video timestamps. URFD videos hold depth and RGB side by
  side; only the RGB half is analysed.
- Perception ran once per video; the tracks were stored and replayed through the
  detectors, so rule variants could be compared on identical inputs. The replay
  reproduces the end-to-end run exactly (44/79 GMDCSA24 falls with the earlier rules,
  both ways).
- Metric: **per video**. A fall video counts as detected when at least one
  `possible_fall` alert (not a near-miss) is raised anywhere in it; an activity (ADL)
  video counts as a false alarm when any alert is raised. Event-level timing and
  latency are not measured here.
- `person_down` never fires on these clips: they are shorter than its 30/60 s limits.

| Dataset | Fall videos | ADL videos | ADL duration (approx.) |
|---|---|---|---|
| GMDCSA24 (4 subjects) | 79 | 81 | 12.5 min |
| UR Fall Detection (cam0) | 30 | 40 | 4.8 min |
| CAUCAFall (10 subjects) | 50 | 50 | 8.4 min |

## Results

| Rules | GMDCSA24 falls | URFD falls | CAUCAFall falls | ADL videos alerted (all) |
|---|---|---|---|---|
| Fast hip drop only | 44/79 (56 %) | 11/30 (37 %) | 11/50 (22 %) | 6/171 |
| **Defaults** (+ quick upright→lying, + box collapse) | **65/79 (82 %)** | **14/30 (47 %)** | **22/50 (44 %)** | **9/171** |

ADL false alarms per dataset with the defaults: GMDCSA24 5/81 (mostly sitting on a bed
and lying down within a second), URFD 1/40, CAUCAFall 3/50. The runs use no zones; on
site, a safe zone over the bed turns those lie-downs into near-misses
(ADR-0012, ADR-0014), which these numbers do not measure. The ADL material is only
about 26 minutes in total, so no false-alarms-per-hour figure is given.

Variants tried on the same data: before the box-collapse rule, lowering the hip-speed
threshold to 0.35–0.45 heights/s or the descent depth to 0.25 heights gained 0–4 falls
per dataset and added up to 3 ADL false alarms. A collapse rule without the minimum hip
drop reached 70/79 on GMDCSA24 but doubled its ADL false alarms (10/81), so it was not
adopted.

## Why falls were missed (defaults)

| Cause | GMDCSA24 | URFD | CAUCAFall |
|---|---|---|---|
| Never looks lying in 2D (fall towards or away from the camera) | 1 | 1 | 15 |
| Lying only in untracked, weak detections | 1 | 8 | 3 |
| Wide box, but the pose does not look lying | 2 | 1 | 3 |
| Seen lying, no rule fired | 10 | 1 | 6 |
| Suspicion raised, clip ends before the rule completes | 0 | 5 | 1 |

What this means:

- Falls along the camera axis are the largest gap. They need depth cues or a learned
  temporal classifier (post-MVP), not more tuning of 2D geometry.
- Falls seen only as weak, untracked detections are left to the floor rule, after its
  limit.
- Several URFD clips end less than a second after the suspicion; on a live stream the
  rule would complete.
- On site, place the camera so that the main floor areas are seen from the side rather
  than along the line of sight.

## Reproduce

Download the datasets yourself (see [`../DATASETS.md`](../DATASETS.md)) and fetch the
models (`fim models fetch`). `fim analyze --source <video>` writes the `events.jsonl` used
by this method for one video. A dataset-level command (`fim eval`) is planned with the
evaluation work; until then, per-video results are aggregated as described above.
