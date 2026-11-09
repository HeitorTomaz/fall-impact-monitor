# ADR-0011: Modular monolith with a reusable detection core

- Status: Accepted
- Date: 2026-11-08

## Context

The reference installation needs one dependable deployable that captures, records, detects, stores and notifies. Other people may want only the fall-detection intelligence, for different purposes: offline analysis of video files, integration into their own alerting, or experimentation. Microservices and message brokers would add operational cost without a measured need.

## Decision

- Ship **one application** (the `app` service, package `fall_impact_monitor`) structured as a **modular monolith**.
- The **detection core** consists of the `api`, `perception` and `analysis` packages:
  - `api` is a small public facade: it receives frames with timestamps (and optional audio) and returns events;
  - the core may depend on numerical and inference libraries (NumPy, SciPy, ONNX Runtime, PyAV), but **must not import** the database, Telegram, webhook, recording or health packages, nor their third-party libraries.
- The core is installable without application dependencies: application packages (aiogram, SQLAlchemy, database driver, Alembic) live in an **extra dependency group** in `pyproject.toml`.
- **Layering is enforced by an automated check** in CI (for example an import-contract tool or a dedicated test).
- The CLI `fim analyze` runs the core on a **video file** and writes events as JSON; the evaluation harness uses the same path.
- **Detectors and sinks are toggled by environment variables** (ADR-0007): the system runs with Telegram disabled, with only the webhook or the JSON log, or with a subset of detectors.
- Vendored third-party code lives in `_vendor/` and is used through the core's own interfaces.

## Alternatives considered

- **Separate services for capture, inference and notification**: process isolation, but more containers, inter-service protocols and failure modes; rejected for now.
- **Separate repositories for the core and the application**: cleaner packaging, but slows a single maintainer; a single repository with an extra dependency group gives most of the benefit.

## Consequences

- The detection logic is testable without cameras, databases or network.
- Adopters can reuse detection through the Python API, the CLI or the event sinks.
- Future integrations (MQTT with Home Assistant discovery, publishing the core on PyPI) are planned after the MVP and need no restructuring.
- Keeping the facade small and stable becomes a review concern for every pull request touching the core.

## References

- https://import-linter.readthedocs.io/
