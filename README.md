# fall-impact-monitor

Open-source, privacy-first detection of possible falls and possible head impacts using local computer vision and IP cameras.

> **Safety notice.** This is experimental assistive software for residential use. It is not a medical device, never diagnoses injury and does not replace human supervision or emergency services.

**Status:** early development (capture and timing diagnostics). Design: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md), decisions in [`docs/adr/`](docs/adr/), contributor conventions in [`CLAUDE.md`](CLAUDE.md).

## Quick start (Docker)

Requirements: Docker Engine with the Compose plugin (Docker Desktop works for development).

```bash
cp .env.example .env                       # then edit .env
docker compose up -d                       # MediaMTX relay and recorder
docker compose run --rm app fim --help     # tools (the monitoring service comes later)
```

Everything runs in containers. Images are built locally; none are published.

## Development

```bash
docker compose --profile dev build dev
docker compose --profile dev run --rm dev                                  # tests
docker compose --profile dev run --rm dev uv run --frozen ruff check
docker compose --profile dev run --rm dev uv run --frozen ruff format --check
docker compose --profile dev run --rm dev uv run --frozen mypy
```

Native development with [uv](https://docs.astral.sh/uv/) (`uv sync && uv run pytest`) also works but is optional.

### Simulated camera

Without a camera, the `sim` profile loops a local video file into MediaMTX as `rtsp://mediamtx:8554/cam1`. Comment out `MTX_PATHS_CAM1_SOURCE` in `.env` first: a path with a camera source does not accept a publisher.

```bash
SIM_FILE=path/inside/test_videos.mp4 docker compose --profile sim up -d mediamtx camera-sim
```

`test_videos/` is git-ignored; never commit recordings.

## Configuration

All settings are environment variables, normally set in `.env` (see `.env.example`).

| Variable | Default | Description |
|---|---|---|
| `FIM_LOG_LEVEL` | `INFO` | Log level (`DEBUG`, `INFO`, `WARNING`, `ERROR`). |
| `FIM_CAMERAS` | *(empty)* | Comma-separated camera ids (`[a-z0-9_]+`). |
| `FIM_CAMERA_<ID>_MAIN_URL` | *(required per camera)* | Main stream URL of camera `<ID>` (upper-case id). Credentials are masked in logs. |
| `FIM_CAMERA_<ID>_SUB_URL` | *(none)* | Optional substream used for analysis. |
| `FIM_STALE_TIMEOUT_S` | `10` | Seconds without frames before a stream is reported stale and reconnected. |
| `FIM_PACKET_BUFFER_S` | `20` | Seconds of encoded video kept in memory per camera for re-analysis. |
| `FIM_MEDIAMTX_API_URL` | `http://mediamtx:9996` | MediaMTX playback server used to export clips. |
| `FIM_MODELS` (build argument) | `yolox_s,rtmpose_s` | Models from `models.toml` downloaded and verified when the app image is built (`docker compose build --build-arg FIM_MODELS=...`). |
| `MTX_PATHS_<NAME>_SOURCE` | *(none)* | Read by MediaMTX: source URL of relayed path `<NAME>` (e.g. `CAM1`, `CAM1SUB`). |
| `SIM_MEDIA_DIR` | `./test_videos` | Development only: host folder with videos for the simulated camera. |
| `SIM_FILE` | `sample.mp4` | Development only: video file (relative to `SIM_MEDIA_DIR`) looped by the simulated camera. |
| `SIM_TARGET` | `rtsp://mediamtx:8554/cam1` | Development only: RTSP URL the simulated camera publishes to. |

## Tools

- `fim probe` — records per-frame timing (camera PTS, arrival time, timestamp quality) for a file or RTSP source and writes CSV, JSON and Markdown summaries under `artifacts/`.
- `fim clip` — exports a time range from the MediaMTX recordings as an MP4 file.
- `fim models fetch|verify` — downloads model weights listed in `models.toml` from their original source and verifies their SHA-256.

## License

Apache License 2.0 — see [`LICENSE`](LICENSE), [`NOTICE`](NOTICE) and [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md). Models and datasets are not part of this repository; see [`MODELS.md`](MODELS.md) and [`docs/DATASETS.md`](docs/DATASETS.md).
