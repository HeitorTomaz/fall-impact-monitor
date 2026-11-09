# ADR-0013: MediaMTX configuration generated from the camera variables

- Status: Accepted
- Date: 2026-11-09

## Context

ADR-0001 makes MediaMTX the single RTSP relay and recorder. Until now its paths were declared twice: once in `FIM_CAMERA_<ID>_*` for the app, and again in `MTX_PATHS_<NAME>_SOURCE` for MediaMTX, with a static `mediamtx.yml` listing the paths. Adding a camera meant editing three places, and the two declarations could drift apart. Cameras must be added by configuration only, with one source of truth.

## Decision

- A one-shot Compose service, `mediamtx-config`, runs `fim mediamtx-config` from the app image on every `docker compose up` and writes MediaMTX's configuration from `FIM_CAMERAS` and the per-camera URLs.
- MediaMTX starts after it, with the generated file as its configuration argument.
- Per camera, path `<id>` relays and records the main stream, and path `<id>sub` relays the substream for analysis without recording it. The second path exists only when `FIM_CAMERA_<ID>_SUB_URL` is set.
- The age limit of recordings comes from `FIM_RECORDINGS_MAX_AGE_H` (MediaMTX `recordDeleteAfter`).
- The file is JSON, which MediaMTX reads as YAML, so no YAML library is needed. It contains the camera URLs, so it is written owner-readable only (0600) on an internal volume and never printed.
- The URL value `publisher` makes a path accept a publisher. The simulated camera and the integration tests use it.

## Alternatives considered

- **Keep `MTX_PATHS_*` variables**: two sources of truth for every camera; rejected.
- **MediaMTX control API** (the app adds paths at runtime): the API must be enabled and reachable, and paths disappear when MediaMTX restarts until the app adds them again. This adds a failure mode to recording; rejected.
- **Template the YAML with a shell script in the MediaMTX container**: the official image has no shell utilities to rely on, and validation would be duplicated; rejected.

## Consequences

- Adding a camera means adding variables and running `docker compose up -d`. MediaMTX picks up the new file when it restarts.
- Camera credentials are stored in the generated file inside the `mediamtx-config` volume, which only `mediamtx-config` and `mediamtx` mount.
- The integration tests use a fixture configuration produced by the same generator; a test keeps the fixture in sync.
