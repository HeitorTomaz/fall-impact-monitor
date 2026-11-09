# ADR-0004: Rule-based detection with a confirmation stage

- Status: Accepted
- Date: 2026-11-08

## Context

The system must detect possible falls even when the person gets up immediately, people lying on the floor even when the fall was not seen, and possible head impacts even without a fall. Recall comes first, but false alarms (for example from children playing) cause alert fatigue. No permissively usable, validated fall classifier exists to reuse whole, and a single monocular camera at 20–30 fps cannot measure a head impact, only the motion before and after it. The design budget for alerting is 60 s (target ≤ 30 s), which leaves room to look again before alerting.

## Decision

**Two stages.**

1. **Real-time stage**: the analysis stream produces **candidates** at an adaptive cadence.
2. **Confirmation stage**: for every candidate, the in-memory packet buffer (ADR-0002) is re-decoded at full rate over the event window, pose runs on every frame with a valid timestamp, and the system observes a few seconds after the event (got up, stayed down, staggered, hand to head). Alert target ≤ 30 s after the event; the design budget is 60 s. A late alert is never discarded: it is sent, marked as late, and the delay is reported as degraded monitoring.

**Detectors** (each can be enabled or disabled by configuration):

- **Fall** (`possible_fall`): per-track state machine, adapted from edgefallkit (normal → suspected → fallen → recovering → normal), using vertical hip/shoulder velocity normalized by body height, torso angle and box aspect ratio. A fall alerts **even if the person recovers immediately**; recovery never cancels an event. A `suspected → normal` transition becomes a persisted near-miss.
- **Person down** (`person_down`): a second-level alert when a person stays low for a configurable time after a fall, **and** a separate path when a person is lying outside **allowed-lying zones** (sofa, bed, play mat) for a configurable time with no fall observed (slow slide, occluded fall, person entering the frame already down). Children are not exempt.
- **Possible head impact** (`possible_head_impact`, **experimental**): fusion of independent evidence within a few seconds: (a) abrupt head deceleration or reversal near a user-drawn **hazard zone**; (b) an audio thud; (c) a post-event reaction (stagger, hand to head, sudden sitting, fall). Alert with ≥ 2 evidences (configurable); one evidence is a **near-miss**: persisted, never notified. A fall with the head inside a hazard zone is also tagged as a possible head impact.
- **Confidence score**: every event carries a confidence score from 0.0 to 1.0 (shown as a percentage). An event is notified only when its score reaches the configurable threshold (e.g. `FIM_ALERT_CONFIDENCE_THRESHOLD`, default 0.5); below it, the event is persisted as a near-miss and never notified. For `possible_head_impact`, the ≥ 2 evidences gate applies in addition to the threshold. Rule-based scores are not calibrated probabilities; on-site calibration maps them closer to probabilities and the default may be revised from that data.
- **Kinematics** use only frames with valid timestamps, after One Euro filtering of keypoints. Poor timing lowers evidence confidence; it never fabricates velocities.
- Zones are drawn once per camera at installation and stored as local, ignored files.
- Aggregation of repeated events and the repeat-notification rule are defined in ADR-0006.

## Alternatives considered

- **Alerting directly from the real-time stage**: faster, but lower precision and no post-event observation.
- **Learned skeleton classifiers now**: published weights are trained on data with restrictive terms and generalize poorly from staged datasets; small classifiers trained locally are planned after the MVP, combined with the rules.
- **Single kinematic head-impact detector**: keypoint jitter and motion blur mimic impacts; insufficient on its own.

## Consequences

- Rules are explainable and calibratable with site data; thresholds live in configuration.
- The confirmation stage adds CPU bursts; the scheduler must prioritize it.
- Head-impact output is labelled experimental and phrased as "possible"; it never claims contact or injury.
- Near-misses give data to search for false negatives and to tune thresholds.

## References

- https://github.com/suharvest/edgefallkit
- https://github.com/casiez/OneEuroFilter
