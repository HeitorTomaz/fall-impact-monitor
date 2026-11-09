# ADR-0008: Docker Compose deployment with local builds

- Status: Accepted
- Date: 2026-11-08

## Context

The whole project must run in containers, both to simplify maintenance of a host that is not always physically accessible and to ease adoption by others. The host is a single machine, possibly without internet for long periods; the container runtime is installed manually by the operator. Model weights have heterogeneous licences (ADR-0009) and must not be redistributed by the project.

## Decision

- **Docker Compose** is the deployment and development orchestrator, with three services:
  - `mediamtx`: official image, pinned version (ADR-0001);
  - `postgres`: official image, pinned major version (ADR-0005);
  - `app`: built locally from the repository `Dockerfile` (non-root user).
- No other service is added without a new ADR.
- **Local builds only**: the project publishes no images. Adopters clone the repository, copy `.env.example` to `.env`, and run `docker compose up -d`; Compose builds the `app` image on first start.
- **Model weights are downloaded at build time** from their original sources, listed in the model manifest with URL, licence and **SHA-256**, and verified; a mismatch fails the build. Nothing is downloaded at runtime, so the system works **offline** after the build.
- `restart: unless-stopped` and health checks on all services. The `app` health reflects real progress (frames, inference, sinks); an internal watchdog exits the process on a stall so Docker restarts it.
- **No ports published** to the network by default; Telegram uses outbound long polling. LAN RTSP viewing can be enabled explicitly.
- **Updates**: `git pull && docker compose up -d --build`; migrations run at start; volumes keep data. The previous `app` image is kept under a tag for **rollback**, and the README includes an on-site checklist (services healthy, `/status` answers, test alert received).
- Development on Windows uses Docker Desktop; tests and CLI commands run with `docker compose run --rm app ...`. Native execution with uv is optional.

## Alternatives considered

- **Kubernetes / Helm**: unnecessary for a single host; containers stay 12-factor so a chart can be added later.
- **Published images (GHCR)**: convenient, but would redistribute weights and GPL-licensed binaries bundled by some wheels; revisit after the MVP with a licence review.
- **Runtime model download**: breaks offline operation and makes start-up depend on third-party hosts.

## Consequences

- First build needs internet and takes longer; later operation does not.
- Each adopter downloads models themselves, under the terms recorded in `MODELS.md`.
- A broken release can be rolled back on site without network access.

## References

- https://docs.docker.com/compose/
