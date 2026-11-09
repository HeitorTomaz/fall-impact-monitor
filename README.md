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

Without a camera, the `sim` profile loops a local video file into MediaMTX as `rtsp://mediamtx:8554/cam1`. Set `FIM_CAMERA_CAM1_MAIN_URL=publisher` in `.env` first: a path that pulls a camera does not accept a publisher.

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
| `FIM_CAMERA_<ID>_MAIN_URL` | *(required per camera)* | Main stream URL of camera `<ID>` (upper-case id). Credentials are masked in logs. MediaMTX relays and records it as path `<id>`. The value `publisher` (development) makes the path accept a publisher instead, such as the simulated camera. |
| `FIM_CAMERA_<ID>_SUB_URL` | *(none)* | Optional substream used for analysis; relayed (not recorded) as path `<id>sub`. |
| `FIM_CAMERA_<ID>_ZONES_FILE` | *(none)* | Optional zone file of camera `<ID>` (allowed-lying zones, such as a bed or a mattress on the floor: possible falls ending there are stored as near-misses and nobody lying there raises a floor event; hazard zones; see `fim zones`), for the monitoring service; `fim analyze` takes `--zones` instead. Keep it in a git-ignored path such as `local/`. |
| `FIM_STALE_TIMEOUT_S` | `10` | Seconds without frames before a stream is reported stale and reconnected. |
| `FIM_PACKET_BUFFER_S` | `20` | Seconds of encoded video kept in memory per camera for re-analysis. |
| `FIM_MEDIAMTX_API_URL` | `http://mediamtx:9996` | MediaMTX playback server used to export clips. |
| `FIM_MODELS` (build argument) | `yolox_m_humanart,rtmpose_s` | Models from `models.toml` downloaded and verified when the app image is built (`docker compose build --build-arg FIM_MODELS=...`). |
| `FIM_MODELS_DIR` | `models` | Folder with verified model files (`<name>.onnx`). |
| `FIM_DETECTOR_MODEL` | `yolox_m_humanart` | Person detector from `models.toml`. Choose `yolox_s` for permissively licensed training data (see `MODELS.md`). |
| `FIM_DETECTOR_SCORE_THRESHOLD` | `0.1` | Minimum detection score. Every detection above it is reported to the rules, tracked or not. |
| `FIM_NEW_TRACK_MIN_SCORE` | `0.3` | Detections at or above this score start and continue tracks; weaker ones are reported without a track. |
| `FIM_TRACK_LOST_S` | `3` | Seconds a track survives without a matching detection (keeps the id through brief misses). |
| `FIM_PERSON_HOLD_S` | `3` | Seconds the person analysis rate is kept after the last detection. |
| `FIM_POSE_MODEL` | `rtmpose_s` | Pose model from `models.toml` (`rtmpose_m` is more accurate and slower). |
| `FIM_EMPTY_FPS` | `1` | Analysis rate while no person is visible. |
| `FIM_PERSON_FPS` | `5` | Analysis rate while people are visible. |
| `FIM_MIN_ANALYSIS_FPS` | `0.5` | Below this analysis rate (10 s window) a camera is reported as degraded monitoring. |
| `FIM_ALERT_CONFIDENCE_THRESHOLD` | `0.5` | Events with a confidence at or above this value (0–1) are notified; the others are kept as near-misses. Scores are rule-based and uncalibrated until calibrated on site. |
| `FIM_FALL_DETECTOR_ENABLED` | `true` | Detect possible falls, including falls followed by immediate recovery. |
| `FIM_FLOOR_DETECTOR_ENABLED` | `true` | Detect a person lying on the floor outside allowed zones, with or without an observed fall. |
| `FIM_PERSON_DOWN_S` | `30` | Seconds lying outside allowed zones after a possible fall before a "person down" event. |
| `FIM_FLOOR_NO_FALL_S` | `60` | Seconds lying outside allowed zones, without an observed fall, before a "person down" event. |
| `FIM_RECORDINGS_MAX_AGE_H` | `168` | Hours a continuous recording segment is kept (MediaMTX deletes older ones). |
| `FIM_RECORDINGS_MAX_GB` | `200` | Disk cap for continuous recordings; `fim recordings retention` deletes the oldest segments above it. Size it to the server's disk. |
| `FIM_RECORDINGS_DISK_MIN_FREE_GB` | `10` | Free space kept on the recordings disk by deleting the oldest recordings; below it after cleanup, disk pressure is reported. |
| `FIM_RECORDINGS_DIR` | `/recordings` | Recordings folder inside the app container (the `recordings` volume). |
| `FIM_EVENT_CLIP_DIR` | `/events` | Event clips folder inside the app container (the `events` volume). |
| `FIM_EVENT_CLIP_PRE_S` | `20` | Seconds of video before the event start in an event clip. |
| `FIM_EVENT_CLIP_POST_S` | `60` | Seconds of video after the event end in an event clip. |
| `FIM_EVENT_CLIP_MAX_MB` | `50` | Maximum clip size (the Telegram limit); larger clips are re-encoded at a lower bitrate. |
| `FIM_EVENT_CLIP_RETENTION_DAYS` | `30` | Days an event clip is kept. Event clips are never deleted for disk pressure. |
| `SIM_MEDIA_DIR` | `./test_videos` | Development only: host folder with videos for the simulated camera. |
| `SIM_FILE` | `sample.mp4` | Development only: video file (relative to `SIM_MEDIA_DIR`) looped by the simulated camera. |
| `SIM_TARGET` | `rtsp://mediamtx:8554/cam1` | Development only: RTSP URL the simulated camera publishes to. |

## Tools

- `fim probe` — records per-frame timing (camera PTS, arrival time, timestamp quality) for a file or RTSP source and writes CSV, JSON and Markdown summaries under `artifacts/`.
- `fim clip` — exports a time range from the MediaMTX recordings as an MP4 file.
- `fim analyze` — runs detection, tracking, pose and the detectors on a file or stream; writes `tracks.jsonl`, `events.jsonl` and, with `--overlay`, an annotated video under `artifacts/` (local inspection only). `--zones` applies a zone file.
- `fim zones` — writes a snapshot and a zone-file template (normalized polygons) to edit by hand; `--check` validates a zone file. See [`docs/fall-detection.md`](docs/fall-detection.md).
- `fim benchmark detectors` — person-detector recall on fallen and upright people using locally downloaded CAUCAFall videos.
- `fim models fetch|verify` — downloads model weights listed in `models.toml` from their original source and verifies their SHA-256.

## License

Apache License 2.0 — see [`LICENSE`](LICENSE), [`NOTICE`](NOTICE) and [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md). Models and datasets are not part of this repository; see [`MODELS.md`](MODELS.md) and [`docs/DATASETS.md`](docs/DATASETS.md).
