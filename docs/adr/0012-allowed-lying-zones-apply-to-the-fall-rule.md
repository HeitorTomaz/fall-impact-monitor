# ADR-0012: Allowed-lying zones apply to the fall rule

- Status: Superseded by ADR-0014
- Date: 2026-11-09

## Context

ADR-0004 defines allowed-lying zones (bed, sofa, play mat) for the person-on-the-floor rule only. Public-dataset results showed that the fall rule alerts when someone sits on a bed and lies down within about a second: the hip drops and the body turns horizontal, exactly as in a fall. In a home this happens every night, so these alerts would be frequent false alarms and would erode trust in the alert group. A bed can also be a mattress on the floor, which no geometric rule can tell apart from a person lying on the floor. Falling out of bed onto the floor, however, must still alert.

## Decision

- The same allowed-lying zones apply to both rules. A person is inside a zone when the centre of their box is inside the zone polygon.
- A `possible_fall` whose person is inside an allowed-lying zone when the event fires (including a track lost during a descent) is **persisted as a near-miss** with the zone name, and never notified. The fall state machine returns to `normal`, so a later fall can still be raised.
- A person seen **lying inside an allowed-lying zone and then lying outside every zone** (within 3 s, for at least the 0.8 s lying confirmation) arms a fall suspicion, even without a measurable drop: falling or rolling out of bed onto the floor alerts.
- A mattress on the floor is handled by drawing it as an allowed-lying zone.
- The person-on-the-floor rule is unchanged: lying outside every allowed-lying zone for its limit still raises `person_down`.

## Alternatives considered

- **Keep zones out of the fall rule** (recall first): every quick lie-down in bed would alert; rejected by the maintainer as too many false alarms.
- **Suppress in-zone events without persisting them**: loses the data needed to review missed events; rejected (near-misses are always persisted).
- **Separate zone lists for each rule**: more configuration for the installer with no identified need; rejected.

## Consequences

- A real fall that lands inside an allowed-lying zone (onto the bed or the sofa) is not notified. This is the accepted cost; the near-miss remains available for review.
- Zone polygons must cover only the lying surface. In the image, floor next to a bed must stay outside the polygon, or a fall onto that floor is treated as lying in bed.
- A person sliding from the sofa or bed to the floor while playing, or a box that stays outside the zone edge for a second, can raise a possible fall (recall first for the zone exit).
- Children are treated like adults: the zones exempt places, never people (ADR-0010).
