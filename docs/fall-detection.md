# Fall and person-on-the-floor detection

Status: MVP rules. They are rule-based and **not calibrated**: a confidence of 70 % is a
score, not a probability. Calibrate on site data before trusting the numbers.

Every event says "possible". The system never confirms a fall or claims an injury.

## Inputs

The detectors consume analysed frames from the perception pipeline
(`fall_impact_monitor.api`): tracked people with a box, COCO-17 keypoints and an identity
flag (`stable` or `uncertain`), plus every person detection above
`FIM_DETECTOR_SCORE_THRESHOLD`, tracked or not. Time is the monotonic analysis time of
each frame. People are analysed at `FIM_PERSON_FPS` (5 by default), so the rules are
written for a few frames per second.

Per person and frame, `analysis/observation.py` computes:

| Feature | Meaning |
|---|---|
| `height_px` | Box height. |
| `aspect` | Box width / height. |
| `hip_y`, `shoulder_y` | Mean of the visible hip and shoulder keypoints (score ≥ 0.25). |
| `torso_angle_deg` | Angle of the shoulder→hip vector from vertical: 0 = upright, 90 = horizontal. |

An observation without both hips and shoulders is invalid: it can hold a state but never
starts a fall.

## Possible fall (`possible_fall`)

One state machine per tracked person: `normal → suspected → fallen → recovering → normal`
(adapted from edgefallkit, see `THIRD_PARTY_NOTICES.md`).

- **Baseline.** While the person is upright, the hip position and the box height are the
  baseline. Hip motion is measured in baseline person heights, so distance to the camera
  does not matter.
- **Suspected.** A horizontal cue (torso ≥ 55° or aspect ≥ 1.25) within 0.75 s of a fast
  hip drop (≥ 0.6 person heights per second).
- **Event.** While suspected, the hip has dropped ≥ 0.35 person heights in a horizontal
  posture, or the person has been lying (both cues) for ≥ 0.8 s. The event fires on the
  descent: **a person who gets up at once still raises it**. Recovery never cancels it.
- **Near-miss.** A suspicion that ends upright, or after 2 s, without an event (a
  stumble, a fast sit-down) is stored as a near-miss and never notified.
- **Track lost during a descent.** If the person stops being tracked while suspected
  (common when someone falls at the edge of the view and is only weakly detected), a
  possible fall is raised after 1 s.
- **Cooldown.** 5 s after an event, a new suspicion cannot start for that person.

### Confidence

```
evidence = 0.30 * drop + 0.25 * speed + 0.30 * posture + 0.15 * lying
  drop    = min(1, max_drop_heights / 0.70)
  speed   = min(1, max_drop_speed_hps / 1.2)
  posture = 0.5 * clip((max_torso_angle_deg - 35) / 45) + 0.5 * clip(max_aspect - 1)
  lying   = min(1, lying_s / 2)

confirmed fall:            0.3 + 0.7 * evidence
track lost while falling:  0.2 + 0.7 * evidence
unconfirmed suspicion:           0.7 * evidence   (always a near-miss)
```

Events at or above `FIM_ALERT_CONFIDENCE_THRESHOLD` (0.5) are notified. With these
weights, a confirmed fall with half of the maximum evidence already scores 0.65 (recall
first).

## Person on the floor (`person_down`)

A person lying (torso ≥ 55° and aspect ≥ 1.25; aspect alone when no pose is available)
whose box centre is outside every `allowed_lying` zone:

- after a possible fall of the same track (alert or near-miss, within the last 120 s):
  `FIM_PERSON_DOWN_S` (30 s);
- otherwise, for slow slides, falls that were not seen, or a person entering the view
  already down: `FIM_FLOOR_NO_FALL_S` (60 s).

Untracked detections count too: a lying detection that stays at the same place
(IoU ≥ 0.3) for `FIM_FLOOR_NO_FALL_S` raises the event with no track id. One event per
lying episode; the episode ends when the person is no longer lying or has not been seen
for 3 s. Children are not exempt. Confidence: 0.6 at the limit, rising linearly to 1.0 at
twice the limit.

## Zones

`fim zones --source <file or stream> --out-dir local/zones/cam1` writes `snapshot.png` and
a `zones.json` template. Edit the polygons by hand, in normalized coordinates (0..1,
origin at the top-left of the image), check the file with `fim zones --check <file>` and
reference it with `FIM_CAMERA_<ID>_ZONES_FILE`.

```json
{
  "frame_size": [1280, 720],
  "zones": [
    {"name": "sofa", "kind": "allowed_lying", "polygon": [[0.05, 0.55], [0.35, 0.55], [0.35, 0.9], [0.05, 0.9]]},
    {"name": "wall", "kind": "hazard", "polygon": [[0.9, 0.0], [1.0, 0.0], [1.0, 1.0], [0.9, 1.0]]}
  ]
}
```

`allowed_lying` zones (sofa, bed, play mat) exempt people lying in them from the floor
rule; they do not affect the fall rule. `hazard` zones (walls, furniture edges) are
reserved for the head-impact rules. Zone files describe a household: keep them in
git-ignored paths.

## Calibration procedure

1. Run the system with only the developer in the alert group.
2. Stage falls on a mattress (with and without immediate recovery, several directions and
   distances) and everyday activities (sitting down fast, lying on the sofa, picking an
   object from the floor). Record the time of each.
3. Run `fim analyze` on the recordings and compare `events.jsonl` with the log: count
   detected falls, missed falls and false alarms per hour; look at the near-misses and at
   the confidence of each event.
4. Adjust `FIM_ALERT_CONFIDENCE_THRESHOLD`, `FIM_PERSON_DOWN_S`, `FIM_FLOOR_NO_FALL_S` and
   the zones. Keep recall first: a lower threshold costs false alarms, a higher one costs
   missed events.
5. Keep the measurements in the git-ignored `local/` directory; publish only anonymized
   methods and dataset results.

## Known blind spots

- Slow falls and slides raise no `possible_fall`; they are covered only by the floor rule,
  after its limit.
- A person hidden behind furniture, or outside the view, is not seen at all.
- A person lying close to the camera or at the edge of the frame may be detected weakly;
  the track-lost and untracked-detection rules reduce, but do not remove, this gap.
- A fall that ends sitting (against a wall, on a chair) may look like sitting down: it can
  end as a near-miss.
- Children playing on the floor outside allowed zones raise `person_down` events; this is
  deliberate (no suppression by attribution).
- Dataset results are in [`benchmarks/falls.md`](benchmarks/falls.md). They are not
  clinical accuracy figures.
