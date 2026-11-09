# ADR-0007: Configuration via environment variables

- Status: Accepted
- Date: 2026-11-08

## Context

The system runs in Docker Compose on hosts maintained occasionally, is adopted by others with different cameras and integrations, and must keep credentials (RTSP passwords, bot tokens) out of the repository. Architecture v0.1 proposed storing configuration in the database, which would require an editing interface the MVP does not have.

## Decision

- All main settings are **environment variables** with the `FIM_` prefix, loaded and validated with **pydantic-settings** at start-up. Invalid configuration fails fast with a clear English message.
- **Every variable is documented** in the README (name, meaning, default, example) and listed in `.env.example` with placeholder values.
- Secrets live only in the local `.env` file (ignored by Git) or Docker secrets; they are never logged. RTSP URLs are masked in logs.
- Cameras are declared through variables (a list of camera identifiers plus per-camera stream settings); adding a camera is a configuration change, not a code change. Substream and audio are optional per camera.
- Feature switches are variables too: sinks (e.g. `FIM_TELEGRAM_ENABLED`, `FIM_WEBHOOK_URL`), individual detectors, audio analysis, notification locale, thresholds, retention and quotas.
- Data too large for a variable (for example hazard and allowed-lying zone polygons) lives in local files referenced by a variable and kept out of Git.
- The `mediamtx` service configuration is derived from the same variables (single source of truth).
- Each event stores an immutable **snapshot and hash of the active configuration**, so results are reproducible; the database does not hold editable configuration.

## Alternatives considered

- **YAML configuration files**: readable, but split configuration between files and secrets, and are less natural in Compose; usable later for complex structures if needed.
- **Configuration in the database**: needs a UI and migrations for every setting; rejected for the MVP.

## Consequences

- The README variable table is part of the Definition of Done for any new setting.
- Nested or per-camera settings need a clear naming convention, fixed during Phase 1 and documented.
- Configuration changes take effect on restart (`docker compose up -d`).

## References

- https://docs.pydantic.dev/latest/concepts/pydantic_settings/
- https://12factor.net/config
