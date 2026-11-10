# Fall and person-on-the-floor detection

Status: MVP rules. They are rule-based and **not calibrated**: a confidence of 70 % is a
score, not a probability. Calibrate on site data before trusting the numbers.

Every event says "possible". The system never confirms a fall or claims an injury.

## Inputs

The detectors consume analysed frames from the perception pipeline
(`fall_impact_monitor.api`): tracked people with a box, COCO-17 keypoints and an identity
flag (`stable` or `uncertain`), plus every person detection above
`FIM_PERSON_DETECTOR_MIN_SCORE`, tracked or not. Time is the monotonic analysis time of
each frame. People are analysed at `FIM_ANALYSIS_PERSON_VISIBLE_FPS` (5 by default), so the rules are
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

- **Baseline.** The highest hip, and the box height of that frame, among the
  non-horizontal frames of the last 1.5 s (the last one is kept when frames are sparse).
  A frame caught mid-descent therefore does not lower the reference. Hip motion is
  measured in baseline person heights, so distance to the camera does not matter.
- **Suspected.** A horizontal cue (torso ≥ 55° or aspect ≥ 1.25) within 0.75 s of a fast
  hip drop (≥ 0.6 person heights per second). When consecutive analysed frames are more
  than 0.75 s apart (low analysis rate), the speed cannot be measured: a horizontal frame
  whose hip is already ≥ 0.35 person heights below the baseline counts as a fast drop.
  Two more cues arm suspicion when the person is lying (both cues) and the hip is
  ≥ 0.15 person heights below the baseline: the last upright frame is at most 1.5 s old
  (a quick change from upright to lying), or the box height shrank by ≥ 40 % within 1 s
  (a collapse). The collapse also works for a track confirmed mid-fall, without any
  upright frame: the tallest recent frame then becomes the baseline. These cues catch
  falls seen by a camera high up, where perspective makes the hip speed look slow.
- **Event.** While suspected, the hip has dropped ≥ 0.35 person heights in a horizontal
  posture, or the person has been lying (both cues) for ≥ 0.8 s. The event fires on the
  descent: **a person who gets up at once still raises it**. Recovery never cancels it.
- **Suspicion without the rule.** A suspicion that ends upright, or after 2 s, without
  the descent rule (a stumble, a fast sit-down) gets a lower score (below). It is a
  near-miss unless that score reaches the alert threshold.
- **Pose lost.** Without hips and shoulders a fall never starts, but a suspicion still
  ends at the 2 s timeout: a horizontal box counts as the fall, anything else as a
  suspicion without the rule.
- **New track id.** When a new track appears within one person height of where a person
  was lost less than 3 s before, it continues that person's state machine (trackers often
  change the id of a falling person).
- **Track lost during a descent.** If the person stops being tracked while suspected
  (common when someone falls at the edge of the view and is only weakly detected), a
  possible fall is raised after 1 s; its base score keeps it above the default
  threshold.
- **Cooldown.** 5 s after an event, a new suspicion cannot start for that person.
- **Safe zones.** A possible fall whose person's box centre is inside a
  `safe` zone when the event fires (lying down quickly in bed) is stored as a
  near-miss with the zone name and never notified; the state machine returns to normal.
  A person seen lying inside such a zone and then lying outside every zone (within 3 s,
  for at least 0.8 s) arms a suspicion even without a measurable drop, so falling or
  rolling out of bed onto the floor alerts (`left_safe_zone` in the evidence). See
  ADR-0012 and ADR-0014.

### Confidence

```
evidence = 0.30 * drop + 0.25 * speed + 0.30 * posture + 0.15 * lying
  drop    = min(1, max_drop_heights / 0.70)
  speed   = min(1, max_drop_speed_hps / 1.2)
  posture = 0.5 * clip((max_torso_angle_deg - 35) / 45) + 0.5 * clip(max_aspect - 1)
  lying   = min(1, lying_s / 2)

descent rule met:           0.3 + 0.7 * evidence
track lost while falling:   0.4 + 0.7 * evidence
suspicion without the rule:       0.7 * evidence
```

Every event at or above `FIM_ALERT_MIN_CONFIDENCE` (0.5) is notified; the others are
near-misses. With these weights, a descent with half of the maximum evidence already
scores 0.65 (recall first).

## Person on the floor (`person_down`)

A person lying (torso ≥ 55° and aspect ≥ 1.25; aspect alone when no pose is available)
whose box centre is outside every `safe` zone:

- after a possible fall of the same track (alert or near-miss, within the last 120 s):
  `FIM_PERSON_DOWN_DETECTOR_AFTER_FALL_S` (30 s);
- otherwise, for slow slides, falls that were not seen, or a person entering the view
  already down: `FIM_PERSON_DOWN_DETECTOR_WITHOUT_FALL_S` (60 s).

A lying episode belongs to a place, not to a track id: tracked people and untracked
detections lying at the same place (IoU ≥ 0.3) continue the same episode, so id switches,
tracked/untracked flicker and single frames that do not look lying do not restart the
timer. The episode ends when nobody has been seen lying there for 3 s. An observed fall of
any track id seen in the episode applies the shorter limit. Children are not exempt.

Confidence: 0.6 at the limit, rising linearly to 1.0 at twice the limit. One alert per
episode. If the threshold is above the current score, the episode is stored once as a
near-miss and alerts as soon as its score reaches the threshold (at twice the limit at
the latest).

## Zones

Safe zones are set once, at installation:

1. With the stack running, take a snapshot and the suggestions. Run as yourself (`--user`)
   so that you can edit the files afterwards:

   ```bash
   docker compose run --rm --user "$(id -u):$(id -g)" app \
     fim zones --source rtsp://mediamtx:8554/cam1 --out-dir local/zones/cam1
   ```

   This writes `snapshot.png`, `zones.json` and `zones.png` in `local/zones/cam1/`. Beds
   and sofas detected in the snapshot with a score of at least 0.5 (COCO-trained
   `yolox_s`, see `MODELS.md`) become **suggested** zones (`"kind": "suggested_safe"`)
   named `bed_1`, `sofa_1`, …, each the detected box shrunk by 10 % on each side towards
   its centre. Without the model or without detections the file has no zones. An existing
   `zones.json` is never overwritten.
2. Edit the zones with the mouse:

   ```bash
   docker compose run --rm --user "$(id -u):$(id -g)" -p 127.0.0.1:8090:8090 app      fim zones --edit local/zones/cam1/zones.json --host 0.0.0.0
   ```

   Open <http://127.0.0.1:8090/> on the server, or from a laptop through an SSH tunnel
   (`ssh -L 8090:127.0.0.1:8090 <server>`). The port is published on the server's loopback
   only; the page loads nothing from outside. A box around a bed or a sofa also covers
   floor next to it and the space underneath, and a fall onto that floor must alert:

   - drag the points to trim each zone to the soft surface (Shift+click adds a point to
     the selected zone, Alt+click removes one);
   - **New zone** for what the detector cannot know (a mattress on the floor, a play mat):
     click its corners, then double-click;
   - **Accept** each suggestion you checked (its kind becomes `safe`), **Delete** wrong ones;
   - **Save** validates the file like the service does and redraws `zones.png`.

   Stop the editor with Ctrl+C. Editing `zones.json` by hand (normalized coordinates 0..1,
   origin at the top-left) works too. The service refuses a file that still has suggested
   zones.
3. Validate the file (the editor's Save already does it); repeat 2 and 3 until the picture is right:

   ```bash
   docker compose run --rm --user "$(id -u):$(id -g)" app \
     fim zones --check local/zones/cam1/zones.json
   ```

4. Set `FIM_CAMERA_CAM1_ZONES_FILE=local/zones/cam1/zones.json` and restart the service.

If `fim zones` says it cannot write the folder, Docker created `local/zones` before you
did (it then belongs to root): run `sudo chown -R "$(id -u):$(id -g)" local/zones`. Keep
the files readable by everyone: the service reads them as another user.

Zones are not re-detected while the service runs: run `fim zones` again after moving a
bed or a sofa (delete or rename the old `zones.json` first). After an update, run
`fim zones --check` on every zone file: the service stops at start-up on an invalid one.

```json
{
  "frame_size": [1280, 720],
  "zones": [
    {"name": "sofa", "kind": "safe", "polygon": [[0.05, 0.55], [0.35, 0.55], [0.35, 0.9], [0.05, 0.9]]},
    {"name": "bed", "kind": "safe", "polygon": [[0.6, 0.5], [0.95, 0.5], [0.95, 0.9], [0.6, 0.9]]}
  ]
}
```

`safe` zones mark soft places (sofa, bed, a mattress on the floor, play mat). They exempt
people lying in them from the floor rule, turn a possible fall that ends in them into a
near-miss (see above), and a fast head stop inside them is not head-impact evidence
([`head-impact.md`](head-impact.md)). Walls and furniture are not drawn: everything
outside safe zones is treated as a place where a head can hit something. Draw safe zones
over the soft surface only: floor next to a bed must stay outside the polygon in the
image, or a fall onto it is treated as lying in bed. Zone files from earlier versions are
rejected at start-up: rename `allowed_lying` to `safe` and delete `hazard` zones
(ADR-0014). Zone files describe a household: keep them in git-ignored paths.

## Calibration procedure

1. Run the system with only the developer in the alert group.
2. Stage falls on a mattress (with and without immediate recovery, several directions and
   distances) and everyday activities (sitting down fast, lying on the sofa, picking an
   object from the floor). Record the time of each.
3. Run `fim analyze` on the recordings and compare `events.jsonl` with the log: count
   detected falls, missed falls and false alarms per hour; look at the near-misses and at
   the confidence of each event.
4. Adjust `FIM_ALERT_MIN_CONFIDENCE`, `FIM_PERSON_DOWN_DETECTOR_AFTER_FALL_S`, `FIM_PERSON_DOWN_DETECTOR_WITHOUT_FALL_S` and
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
- A person lying along the camera axis (feet towards the camera) has a tall, narrow box
  and does not look lying: neither floor rule nor lying cue applies. Falls towards or
  away from the camera often look upright in 2D during the whole fall and are missed.
- A fall that lands inside a safe zone (onto the bed or the sofa) is only a
  near-miss. Lying down quickly on a bed or sofa that has no safe zone can raise
  a possible fall.
- Sliding from a bed or sofa to the floor while lying (children playing) raises a
  possible fall.
- At low analysis rates (≈ 1 fps, under CPU pressure) lying down quickly on the floor can
  raise a possible fall: speed cannot be measured, so recall wins.
- Children playing on the floor outside safe zones raise `person_down` events; this is
  deliberate (no suppression by attribution).
- Dataset results are in [`benchmarks/falls.md`](benchmarks/falls.md). They are not
  clinical accuracy figures.
