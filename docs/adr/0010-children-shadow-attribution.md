# ADR-0010: Children: shadow attribution and opt-in handling

- Status: Accepted
- Date: 2026-11-08

## Context

Children often run, jump, play on the floor and fall without needing the same response as the monitored adult, so they are a major source of false alarms. However, telling children from adults without faces is unreliable: body-proportion and height cues fail exactly when people sit, crouch, lie down or are occluded, which is when falls happen. A tracker identity switch can also transfer a "child" label to an adult. Missing an adult event is worse than a false alarm.

## Decision

- **Never suppress or downgrade an event because a child is present or because attribution is uncertain.** An uncertain person is treated as an adult.
- In the MVP, adult/child/uncertain attribution runs **in shadow mode only**: it is computed and logged with each event for later validation, and it does not change alerts, routing or analysis cadence.
- Attribution, when computed, accumulates over many frames while the person is **upright and fully visible**, and is kept only while the track identity is stable; any crossing, occlusion or identity switch resets it to uncertain. Labels computed during abnormal postures are not used.
- No face recognition, no face-based age estimation and no persistent biometric identity.
- **Opt-in child handling (post-MVP, maintainer decision).** The installer may choose how events attributed to a child are notified, through environment variables (names illustrative):
  - `FIM_CHILD_EVENTS`: `alert` (**default**, same as adults), `silent` (delivered without sound) or `ignore` (not notified);
  - `FIM_CHILD_ATTRIBUTION_MIN_CONFIDENCE`: minimum confidence of the "child" attribution for the chosen mode to apply.
- Guardrails that hold in every mode:
  - an uncertain person is treated as an adult;
  - an identity switch, a crossing/occlusion with another person, or any adult involved in the event cancels `silent`/`ignore` for that event;
  - an ignored event is still persisted with the reason, so it can be reviewed when searching for false negatives;
  - only notification changes: the person keeps being tracked and analyzed, and is re-evaluated continuously;
  - the README documents the risk: an adult mislabelled as a child would not be notified in `ignore` mode.
- Separately, under CPU pressure, confirmed-child tracks may be analyzed at a **reduced cadence, never zero**, returning to full rate on any fall sign or identity doubt — only after validation on site data.
- Each change is recorded in a new ADR with the validation evidence.

## Alternatives considered

- **Height or size filtering**: perspective-dependent and wrong for seated or lying people; rejected as a filter.
- **Skipping children's tracks to save CPU**: an adult mislabelled as a child would go unanalyzed; rejected.
- **Manual "children present" switch that mutes all alerts**: mutes adults too; rejected. The per-event, attribution-based opt-in above was chosen instead, defaulting to `alert`.

## Consequences

- During the MVP, play sessions can generate alerts; aggregation (ADR-0006) and calibration reduce their volume.
- Real installations produce the data needed for validation quickly.
- Regulatory note: inferring age is listed as biometric categorisation under the EU AI Act; the feature stays shadow-only and requires review before being enabled in distributed configurations.

## References

- https://ai-act-service-desk.ec.europa.eu/en/ai-act/recital-16
