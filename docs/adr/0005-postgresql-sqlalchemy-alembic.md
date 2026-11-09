# ADR-0005: PostgreSQL with SQLAlchemy and Alembic

- Status: Accepted
- Date: 2026-11-08

## Context

The application stores structured state: events and their evidence, notification attempts, feedback (seen / false alarm / confirmed), manual missed-event reports, near-misses, the recording-segment index used by retention, camera health and a configuration snapshot per event. Volume is small (one host, a few cameras). Architecture v0.1 proposed PostgreSQL; issue #5 mentioned SQLite.

During the v0.2 review, SQLite in WAL mode was recommended for the MVP as the simpler option (single file, no extra service, no "database unavailable" failure mode). The maintainer chose PostgreSQL, valuing a standard server database for future multi-process and multi-tool access.

## Decision

- Use **PostgreSQL** as the `postgres` service (official image, pinned major version, dedicated volume, no published port).
- Access it through **SQLAlchemy 2.x** with an async-capable driver (asyncpg preferred; psycopg 3 acceptable).
- Manage the schema with **Alembic**; migrations run automatically at `app` start-up and are reviewed in pull requests.
- **No media in the database**: video, audio, images and model files stay on volumes; the database holds paths, checksums and windows.
- Timestamps are stored as UTC `timestamptz`, alongside the source/monotonic values when relevant.
- **The alert path does not depend on the database.** Events are delivered to sinks first; persistence retries in the background. Pending alerts must survive a restart when both the database and the internet are down (for example via a small local outbox file); the mechanism is an implementation choice.
- **Backup is a documented manual procedure** (`docker compose exec postgres pg_dump` / `pg_restore`), tested at least once; automation comes later.

## Alternatives considered

- **SQLite (WAL)**: simpler and sufficient at this scale; not chosen by the maintainer. SQLAlchemy and Alembic keep a migration path open in either direction.
- **Storing configuration in the database**: rejected; configuration comes from environment variables (ADR-0007), and the database keeps only an immutable snapshot per event.

## Consequences

- One more service with a health check, a volume, version upgrades and a backup procedure.
- Integration tests in CI need a PostgreSQL service container; unit tests do not.
- "Database unavailable" is a real failure mode: it is reported as degraded monitoring and must be covered by a test proving alerts still reach the sinks.

## References

- https://docs.sqlalchemy.org/en/20/
- https://alembic.sqlalchemy.org/
- https://www.postgresql.org/docs/current/app-pgdump.html
