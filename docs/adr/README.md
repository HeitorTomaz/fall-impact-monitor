# Architecture Decision Records

This directory records the significant architectural decisions of Fall Impact Monitor.
Each record explains the context, the decision, the alternatives that were considered and the consequences.
The living architecture description is [`docs/ARCHITECTURE.md`](../ARCHITECTURE.md).

## Rules

- One decision per file, named `NNNN-short-title.md` with a four-digit sequence number.
- Write an ADR for any change to the technology stack, Docker services, safety rules, data/model policy or public interfaces.
- **Accepted ADRs are superseded by new ADRs, never rewritten.** Fix typos if needed, but change a decision only by adding a new ADR that references the old one, and set the old one's status to `Superseded by ADR-NNNN`.
- ADRs are public: no household details, credentials, private recordings or measurements taken from private footage.

## Index

| Number | Title | Status |
|---|---|---|
| [0001](0001-mediamtx-relay-and-recorder.md) | MediaMTX as RTSP relay and recorder | Accepted |
| [0002](0002-pyav-capture-and-time-domains.md) | PyAV capture and explicit time domains | Accepted |
| [0003](0003-perception-stack.md) | Perception stack: benchmark-selected detector, OC-SORT, RTMPose on ONNX Runtime | Accepted |
| [0004](0004-detection-with-confirmation-stage.md) | Rule-based detection with a confirmation stage | Accepted |
| [0005](0005-postgresql-sqlalchemy-alembic.md) | PostgreSQL with SQLAlchemy and Alembic | Accepted |
| [0006](0006-optional-event-sinks.md) | Optional event sinks: Telegram, webhook and JSON log | Accepted |
| [0007](0007-configuration-via-environment-variables.md) | Configuration via environment variables | Accepted |
| [0008](0008-docker-compose-local-builds.md) | Docker Compose deployment with local builds | Accepted |
| [0009](0009-model-and-dataset-policy.md) | Model and dataset policy | Accepted |
| [0010](0010-children-shadow-attribution.md) | Children: shadow attribution and opt-in handling | Accepted |
| [0011](0011-modular-monolith-and-detection-core.md) | Modular monolith with a reusable detection core | Accepted |
| [0012](0012-allowed-lying-zones-apply-to-the-fall-rule.md) | Allowed-lying zones apply to the fall rule | Accepted |

## Template

```markdown
# ADR-NNNN: <Title>

- Status: Proposed | Accepted | Superseded by ADR-NNNN
- Date: YYYY-MM-DD

## Context

What problem or force makes a decision necessary.

## Decision

What we decided, stated precisely.

## Alternatives considered

Each serious alternative and why it was not chosen.

## Consequences

What becomes easier, harder or riskier; follow-up work.

## References

Links to sources, issues or related ADRs.
```
