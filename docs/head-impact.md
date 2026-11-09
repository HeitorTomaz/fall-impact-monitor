# Possible head impact (experimental)

**One camera cannot see a head hitting something.** At 5–25 frames per second it sees
the motion before and after a contact, and a microphone hears a sound. Every
`possible_head_impact` event is therefore labelled experimental, says "possible", and
never claims a contact or an injury. Scores are rule-based and uncalibrated.

## Setup

Draw **hazard zones** (walls, furniture edges, door frames, the bathroom sink) with
`fim zones` (see [`fall-detection.md`](fall-detection.md#zones)). Without hazard zones
there is no head-impact detection. Keep the zones around the hazards themselves: a
zone covering the whole room turns every quick movement into evidence.

**Audio** is on by default (`FIM_AUDIO_ENABLED`), but only listened to for cameras
whose zone file has hazard zones, while the head-impact rules are on. It is analysed
locally, in memory,
for sudden sounds only; nothing is recorded by this detector or sent anywhere (the
continuous recordings keep the camera's audio, as before). Tell everyone in the
monitored space, including visitors, that the camera listens.

## Evidence

| Kind | What | Strength | Weight |
|---|---|---|---|
| `head_stop_near_hazard` | The head (mean of nose, eyes and ears; else a point a tenth of the box below its top) moved at ≥ 1 person height per second within the last 0.5 s, then stopped (≤ 30 % of that speed) or reversed, inside a hazard zone or within one head size (0.15 person heights) of it | peak speed / 2 heights per second, at most 1 | 0.5 |
| `thump` | A 32 ms audio frame ≥ `FIM_AUDIO_THUMP_MIN_DB` (12 dB) above the median level of the last second; repeats within 0.3 s count once | rise in dB / (2 × `FIM_AUDIO_THUMP_MIN_DB`), at most 1 | 0.35 |
| `reaction` | A possible fall by the same person (its confidence), or a near-fall (half of it), counted from the time the fall started. A fall that ended in an allowed-lying zone (lying down in bed) counts as a near-fall | as stated | 0.4 |
| `head_in_hazard_zone` | The person's head was in or next to a hazard zone when a possible fall was raised (not for a fall into an allowed-lying zone) | 1 | 0.45 |

Kinematics:

- **Timestamps:** only frames with valid timestamps are used. A zero, backward or
  outlier step breaks the velocity chain, so no speed is computed across it.
- **Units:** positions are smoothed in pixels with a One Euro filter. Speeds are
  divided by the tallest box of the last 2 s, so the distance to the camera does not
  matter, and a box that shrinks (crouching, sitting down, a box cut by furniture) is
  not mistaken for head motion.
- **Analysis rate:** while a head moves near a hazard zone (at least half the
  evidence speed), the pipeline analyses every frame for 2 s instead of the usual 5
  per second. A still head next to a wall does not keep the camera at full rate.

## Fusion

Thresholds:

- **Who opens a candidate:** a candidate opens with the first person-based evidence
  (kinematic or reaction). A thump never opens one: a door or a dropped pan is a thump
  too.
- **Window:** evidence of the same person within `FIM_HEAD_IMPACT_WINDOW_S` (3 s)
  joins the candidate, and so do thumps within 3 s before or after it.
- **Waiting for the fall rule:** the candidate stays open 4 s longer, so a fall
  confirmed a few seconds after it started still joins the head stop it came with. A
  head-impact alert therefore arrives about 7 s after the first evidence.

```
confidence = 1 - Π (1 - weight_k × strength_k)   over the kinds present
```

The candidate alerts only with at least `FIM_HEAD_IMPACT_MIN_EVIDENCE` (2) kinds
**and** a confidence at or above `FIM_ALERT_MIN_CONFIDENCE`. Otherwise it is stored as a
near-miss with its evidence. Examples:

- A strong head stop alone: 0.50, one kind → near-miss.
- A strong head stop with a loud thump: 0.68 → alert.
- A fall (0.8) with the head at a wall: 0.63 → alert.

## Limits and blind spots

- **Hidden contacts:** a contact hidden by the body or furniture, or happening between
  two frames, is not seen. Slow contacts (leaning the head on a wall) are not fast
  enough to count.
- **Camera axis:** heads moving along the camera axis barely change position in the
  image.
- **Look-alikes:** a fast stop near a wall without any contact (stopping just short,
  turning the head) is evidence too; the fusion and the threshold are what keep it
  from alerting alone.
- **Audio:** thumps come from anywhere in the room. Without an audio track, or with
  `FIM_AUDIO_ENABLED=false`, only the head stop and the reaction remain.
- **Not implemented yet** (issue #6 cut order, follow-ups):
  - the PANNs sound classifier;
  - the full-rate re-analysis of the packet buffer before alerting (confirmation
    stage);
  - stagger and hand-to-head reactions.
- **Calibration:** none of the weights is calibrated. Measure staged soft contacts
  (head touching a pillow against a wall, with a thump) and near-contacts on site
  before relying on these alerts (MVP criterion 11, [`pilot.md`](pilot.md)).
