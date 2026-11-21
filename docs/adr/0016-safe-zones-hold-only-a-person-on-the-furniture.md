# ADR-0016: Safe zones hold only a person on the furniture

- Status: Accepted
- Date: 2026-11-20

## Context

A safe zone (ADR-0014) marks a soft place where lying down is normal: a bed, a sofa. A fall that ends **on** that furniture is a near-miss, and a person lying there raises no person-on-the-floor event. A fall that ends on the floor next to the furniture is a fall like any other and must alert.

Two things let the floor next to the furniture count as safe:

- **Membership by the box centre.** A person was in a zone when the centre of their box was. Zones are drawn on the image, and a camera mounted low (about 0.8 m) sees the floor in front of a bed overlapping the bed: a person lying on that floor has the centre of their box over the bed.
- **Suggested zones over the whole furniture box.** `fim zones` (#31) suggested the detected box shrunk by 10 % per side. The box holds the furniture's top, where a person lies, and its front face down to the floor, and sometimes some floor in front of it.

Measured by replay on the train and validation splits of the public datasets (#106), with the zones `fim zones` suggests from the first frame of each video and every suggestion accepted: with the centre test, GMDCSA24 lost 9 of 41 train falls and 6 of 21 validation falls, every one a fall onto the rug in front of a bed.

## Decision

- **Membership:** a person is in a safe zone only when both the centre of their box and the middle of its bottom edge are inside the same zone. It applies to the fall rule (a fall that ends in a zone, leaving a zone) and to the person-on-the-floor rule. The head-impact rules keep judging the head point (ADR-0014).
- **Suggested zones cover the top of the furniture:** before the 10 % shrink, `fim zones` drops the lower part of each detected box, its front face, by a share that depends on the camera's mounting height (`--camera-height-m`, as `FIM_CAMERA_<ID>_HEIGHT_M`, #78): 30 % at 1.0 m or lower, 10 % at 2.2 m or higher, linear in between. The lower the camera, the larger the front face in the image.
- **Evaluation:** `fim eval --safe-zones suggested` analyses each video with the zones `fim zones` would suggest from its first frame, all accepted, and reports it apart from the usual protocol (no zones). Furniture where a dataset's falls happen is left out (the mattress on the floor of the Multiple Cameras Fall Dataset, seen as a bed).

## Alternatives considered

- **Keep the centre test and ask installers to draw tight zones:** a correct zone over a bed's top still overlaps the floor in front of it for a low camera, so the centre test would still hide falls there; rejected.
- **Bottom edge only:** a person sitting on the edge of a bed with their feet on the floor would leave the zone; harmless for recall, but the centre test is kept as well so that a box reaching into a zone from outside (a person standing in front of a sofa) does not count either.
- **A fixed front-face share for every camera:** too much is cut for a high camera, whose zones would no longer cover a person lying on the bed, or too little for a low one; rejected.
- **Estimate the furniture's top from the floor plane (#90):** more precise, but the floor scale of each camera is not learned yet; a follow-up can replace the share.

## Consequences

- **Fewer falls hidden by a zone.** On the train and validation splits with suggested zones, no fall is lost against the protocol without zones (with the centre test 15 were), and 3 false alerts are removed (lying down on a bed or a sofa). On the test split, scored once, results are identical with and without zones.
- **A person on the furniture whose box reaches the floor** (sitting on the edge of a bed, legs down) is outside the zone; a quick drop to that position can alert. Recall first accepts it.
- **The 10 % share for high cameras is a geometric estimate:** the detector found almost no furniture in the high-camera datasets. Installers check every suggestion; the install checklist asks for zones over the top of the furniture only.
- **Existing zone files keep working;** a zone drawn over the whole furniture now holds fewer people, never more.

## References

- Issue #106; ADR-0014 (safe zones), ADR-0012; #31 (suggested zones); #78 (camera height).
