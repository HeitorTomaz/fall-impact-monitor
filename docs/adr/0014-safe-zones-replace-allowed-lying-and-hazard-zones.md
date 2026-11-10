# ADR-0014: Safe zones replace allowed-lying and hazard zones

- Status: Accepted
- Date: 2026-11-09

## Context

ADR-0004 asks the installer to draw two kinds of zone: allowed-lying zones (bed, sofa, play mat) for the fall and floor rules, and hazard zones (walls, furniture edges) for the possible-head-impact rules. ADR-0012 extends allowed-lying zones to the fall rule.

Hazard zones work against the first safety rule:

- **Missed places.** Head-impact evidence exists only where the installer remembered to draw a hazard. A wall or a table edge left out, a piece of furniture moved, or the floor itself (a head hitting the floor during a fall) gives no evidence.
- **Hard setup.** In a home most of the image is a possible hazard. Drawing it all is long and easy to get wrong.

The places where a head is safe are few and easy to name: beds, sofas, a mattress on the floor. They are also the places the allowed-lying zones already mark.

## Decision

- **One zone kind, `safe`,** marks soft places: bed, sofa, a mattress on the floor, play mat. It replaces both `allowed_lying` and `hazard`; the old kinds are rejected at start-up with a message that says what to do. There is no alias.
- **Fall and floor rules:** safe zones do what allowed-lying zones did (ADR-0012), unchanged. The evidence keys become `safe_zone` and `left_safe_zone`.
- **Head-impact rules:**
  - a fast head stop or reversal is evidence **anywhere except inside a safe zone** (judged by the head point);
  - the rules run without any zone file;
  - audio is listened to on every camera while the head-impact rules and `FIM_AUDIO_ENABLED` are on.
- **Evidence kinds** become `head_stop`, `thump` and `reaction`. The `head_in_hazard_zone` kind is removed: without hazard zones it would tag every fall as a possible head impact, even with no fast head motion.
- **Full-rate analysis** starts when a head outside safe zones moves at the head-stop evidence speed (1 person height per second), no longer at half of it. Walking (about 0.6 person heights per second) does not keep a camera at full rate.
- **Recognising walls and furniture automatically** (scene segmentation) is left to a follow-up. It would strengthen the evidence where a hazard is actually visible, never narrow it.

## Alternatives considered

- **Keep hazard zones** (ADR-0004): misses every hazard the installer did not draw; rejected.
- **Keep hazard zones and add safe zones, using hazards only when drawn:** two models to explain and test, for no identified need; rejected.
- **Separate safe zones for the head-impact rules and allowed-lying zones for the fall rules:** a play mat where lying is normal but a head hitting the floor still counts would need it. No such case exists yet, and every extra zone kind is more setup; rejected for now.
- **Recognise walls and furniture with a segmentation model now:** a new model, a license review and a CPU cost on the reference hardware, with no time before the first installation; deferred.

## Consequences

- **More evidence, more near-misses.** A fast head stop anywhere outside safe zones is evidence, so quick play, sport or dancing produce more head-impact near-misses. The rule of at least two evidence kinds and the confidence threshold still keep one kind from alerting alone.
- **Two messages per fall.** Most real falls end with the head stopping fast near the floor. With the fall itself as the reaction, such a fall usually raises both "possible fall" and "possible head impact (experimental)". This is accepted: for a person for whom any head impact matters, it is the case the rules exist for.
- **No head-impact evidence in soft places.** A head hitting a bed frame or a sofa arm inside a safe zone gives none. Safe zones should cover the soft surface only.
- **More CPU for audio.** Every camera with the head-impact rules on reads its audio; tell everyone in the monitored space.
- **Existing zone files need editing:** `allowed_lying` becomes `safe`, and `hazard` zones are deleted.
- **Superseded records:** ADR-0012 (its rules continue under the new zone name), and the hazard-zone parts of ADR-0004.
