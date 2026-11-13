# fall-impact-monitor

Open-source, privacy-first detection of possible falls and possible head impacts using local computer vision and IP cameras.

> **Safety notice.** This is experimental assistive software for residential use. It is not a medical device, never diagnoses injury and does not replace human supervision or emergency services.

**Status:** early development; the monitoring service runs end to end (not yet piloted). Design: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md), decisions in [`docs/adr/`](docs/adr/), contributor conventions in [`CLAUDE.md`](CLAUDE.md).

## Quick start (Docker)

Requirements: Docker Engine with the Compose plugin (Docker Desktop works for development).

```bash
cp .env.example .env                       # then edit .env (cameras, FIM_DB_PASSWORD, Telegram)
mkdir -p local/zones                       # zone files (before the first start, so the folder is yours)
docker compose up -d --build               # monitoring service, MediaMTX relay and recorder, PostgreSQL
docker compose run --rm app fim alert test # check that every enabled output receives a message
docker compose run --rm app fim --help     # tools
```

Then set the safe zones of each camera (beds, sofas): see [`docs/fall-detection.md`](docs/fall-detection.md#zones).

`docker compose up` starts the monitoring service (`fim run`): it reads each camera through the MediaMTX relay, raises events, delivers them to the enabled outputs, exports a clip for each alert and applies retention every 10 minutes. It needs at least one camera in `FIM_CAMERAS`; without one it exits and Docker keeps restarting it. Database migrations run automatically at start-up (retried until PostgreSQL answers; alerts never wait for them).

**Privacy:** Telegram group messages are not end-to-end encrypted. Alerts are text; a video leaves the server only when someone in a configured chat presses Watch video, and then it stays on Telegram's servers. ONNX Runtime's usage telemetry, on by default in its official builds, is turned off (`ORT_DISABLE_TELEMETRY=1`).

Everything runs in containers. Images are built locally; none are published. Installing on a server, updating, rolling back and backing up: [`docs/deployment.md`](docs/deployment.md).

## Known limitations

Read these before relying on the system; in-person supervision remains the primary protection.

- **Not a medical device.** It never diagnoses an injury and does not replace supervision or emergency services. Every alert is a *possible* event.
- **One camera cannot confirm that the head hit something.** "Possible head impact" is experimental.
- **Falls can be missed:**
  - slow falls and slides are only reported as a person on the floor, after a delay;
  - falls towards or away from the camera, people hidden behind furniture, far away or out of view are often missed.
- **Scores are uncalibrated:** confidence scores are rule-based until calibrated on site data ([`docs/pilot.md`](docs/pilot.md)).
- **If the whole server fails** (power, hardware), nobody is notified automatically. `/status` is the manual check: a bot that does not answer means the server is down.
- **Without internet**, detection continues and alerts queue; they arrive, marked late, when the connection returns.
- **Videos sent to Telegram stay on Telegram's servers.** Group chats are not end-to-end encrypted. Videos are sent only when someone presses Watch video.
- **Adult/child attribution does not decide alerts.** Children playing on the floor can raise alerts.

Bringing an installation into service (calibration, fault injection, go-live checklist): [`docs/pilot.md`](docs/pilot.md).

## Development

```bash
docker compose --profile dev build dev
docker compose --profile dev run --rm dev                                  # tests
docker compose --profile dev run --rm dev uv run --frozen ruff check
docker compose --profile dev run --rm dev uv run --frozen ruff format --check
docker compose --profile dev run --rm dev uv run --frozen mypy
```

The `dev` profile starts a throwaway PostgreSQL (`postgres-test`, in memory) for the storage tests. The MediaMTX integration tests need the `itest` profile:

```bash
docker compose --profile itest up -d mediamtx-test
docker compose --profile dev --profile itest run --rm \
  -e FIM_TEST_MEDIAMTX_RTSP=rtsp://mediamtx-test:8554 \
  -e FIM_TEST_MEDIAMTX_PLAYBACK=http://mediamtx-test:9996 \
  dev uv run --frozen pytest tests/integration
```

Native development with [uv](https://docs.astral.sh/uv/) (`uv sync && uv run pytest`) also works but is optional.

### Simulated camera

Without a camera, the `sim` profile loops a local video file into MediaMTX as `rtsp://mediamtx:8554/cam1`. Set `FIM_CAMERA_CAM1_MAIN_URL=publisher` in `.env` first: a path that pulls a camera does not accept a publisher.

```bash
FIM_SIM_FILE=path/inside/test_videos.mp4 docker compose --profile sim up -d mediamtx camera-sim
```

`test_videos/` is git-ignored; never commit recordings.

## Configuration

All settings are environment variables, normally set in `.env` (see `.env.example`).

### General

| Variable | Default | Description |
|---|---|---|
| `FIM_LOG_LEVEL` | `INFO` | Log level (`DEBUG`, `INFO`, `WARNING`, `ERROR`). |

### Cameras

| Variable | Default | Description |
|---|---|---|
| `FIM_CAMERAS` | *(empty)* | Comma-separated camera ids (`[a-z0-9_]+`). |
| `FIM_CAMERA_<ID>_MAIN_URL` | *(required per camera)* | Main stream URL of camera `<ID>` (upper-case id). Credentials are masked in logs. MediaMTX relays and records it as path `<id>`. The value `publisher` (development) makes the path accept a publisher instead, such as the simulated camera. |
| `FIM_CAMERA_<ID>_SUB_URL` | *(none)* | Optional substream used for analysis; relayed (not recorded) as path `<id>sub`. |
| `FIM_CAMERA_<ID>_ZONES_FILE` | *(none)* | Optional zone file of camera `<ID>` with its safe zones: soft places such as a bed, a sofa or a mattress on the floor. Possible falls ending there are stored as near-misses, nobody lying there raises a floor event, and a fast head stop there is not head-impact evidence (see `fim zones`), for the monitoring service; `fim analyze` takes `--zones` instead. Keep it under `local/zones/` (git-ignored, mounted into the app container), e.g. `local/zones/cam1/zones.json`. |
| `FIM_CAMERA_<ID>_ROTATE_DEG` | `0` | `180` for a camera mounted upside down (on a ceiling) whose image is not flipped by the camera itself. Rotates the frames before analysis only; recordings and clips stay as the camera sends them, so prefer the camera's own image-flip setting. When most of the last upright people seen look upside down, the camera is reported as degraded (`image_inverted`) until that changes. `fim zones` and `fim analyze` take `--rotate-deg`; a zone file records the rotation it was drawn with, and the service refuses to start when it differs from this variable (the zones would be mirrored). |

### Capture

| Variable | Default | Description |
|---|---|---|
| `FIM_CAPTURE_STALE_TIMEOUT_S` | `10` | Seconds without frames before a stream is reported stale and reconnected. |
| `FIM_CAPTURE_PACKET_BUFFER_S` | `20` | Seconds of encoded video kept in memory per camera for re-analysis. |
| `FIM_CAPTURE_MAX_ANALYSIS_WIDTH_PX` | `1280` | Frames wider than this are downscaled before analysis (for example a main stream analysed because the camera has no substream). Recordings keep the full resolution. |

### Perception

| Variable | Default | Description |
|---|---|---|
| `FIM_MODELS_DIR` | `models` | Folder with verified model files (`<name>.onnx`). |
| `FIM_MODELS_FETCH` (build argument) | `yolox_m_humanart,rtmpose_s,yolox_s` | Models from `models.toml` downloaded and verified when the app image is built (`docker compose build --build-arg FIM_MODELS_FETCH=...`). |
| `FIM_PERSON_DETECTOR_MODEL` | `yolox_m_humanart` | Person detector from `models.toml`. Choose `yolox_s` for permissively licensed training data (see `MODELS.md`). |
| `FIM_PERSON_DETECTOR_MIN_SCORE` | `0.1` | Minimum detection score. Every detection above it is reported to the rules, tracked or not. |
| `FIM_TRACKER_NEW_TRACK_MIN_SCORE` | `0.3` | Detections at or above this score start and continue tracks; weaker ones are reported without a track. |
| `FIM_TRACKER_LOST_TRACK_TIMEOUT_S` | `3` | Seconds a track survives without a matching detection (keeps the id through brief misses). |
| `FIM_POSE_ESTIMATOR_MODEL` | `rtmpose_s` | Pose model from `models.toml` (`rtmpose_m` is more accurate and slower). |
| `FIM_ANALYSIS_EMPTY_SCENE_FPS` | `3` | Analysis rate while no person is visible. A person who walks in is seen within a third of a second. |
| `FIM_ANALYSIS_PERSON_VISIBLE_FPS` | `8` | Analysis rate while people are visible: a fall lasts about a second. A target: a machine that cannot reach it analyses as fast as it can, always the latest frame (see `docs/benchmarks/pipeline.md`). |
| `FIM_ANALYSIS_PERSON_HOLD_S` | `3` | Seconds the person analysis rate is kept after the last detection. |
| `FIM_ANALYSIS_MIN_FPS` | `0.5` | Below this analysis rate (10 s window) a camera is reported as degraded monitoring. |

### Detectors

| Variable | Default | Description |
|---|---|---|
| `FIM_FALL_DETECTOR_ENABLED` | `true` | Detect possible falls, including falls followed by immediate recovery. |
| `FIM_PERSON_DOWN_DETECTOR_ENABLED` | `true` | Detect a person lying on the floor outside safe zones, with or without an observed fall. |
| `FIM_PERSON_DOWN_DETECTOR_AFTER_FALL_S` | `30` | Seconds lying outside safe zones after a possible fall before a "person down" event. |
| `FIM_PERSON_DOWN_DETECTOR_WITHOUT_FALL_S` | `60` | Seconds lying outside safe zones, without an observed fall, before a "person down" event. |
| `FIM_HEAD_IMPACT_DETECTOR_ENABLED` | `true` | Experimental possible head impact: fuses a fast head stop anywhere outside safe zones, a thump in the audio and a reaction (fall); needs no zone (see [`docs/head-impact.md`](docs/head-impact.md)). |
| `FIM_HEAD_IMPACT_MIN_EVIDENCE` | `2` | Kinds of evidence needed for a head-impact alert; fewer is a near-miss. |
| `FIM_HEAD_IMPACT_WINDOW_S` | `3` | Seconds over which evidence of one person is combined. |
| `FIM_AUDIO_ENABLED` | `true` | Listen to each camera's audio (through the relay) for thumps, while the head-impact rules are on. Inform everyone in the monitored space; audio is never sent anywhere. |
| `FIM_AUDIO_THUMP_MIN_DB` | `12` | A sound this many dB above the background of the last second counts as a thump. |

### Alerts

| Variable | Default | Description |
|---|---|---|
| `FIM_ALERT_MIN_CONFIDENCE` | `0.5` | Events with a confidence at or above this value (0–1) are notified; the others are kept as near-misses. Scores are rule-based and uncalibrated until calibrated on site. |
| `FIM_ALERT_LANGUAGE` | `en` | Language of notification texts: `en` or `pt-BR`. Logs and JSON stay in English. |
| `FIM_ALERT_TIMEZONE` | `UTC` | IANA time zone of the times shown in notifications, e.g. `America/Sao_Paulo`. Stored times are always UTC. |
| `FIM_ALERT_AGGREGATION_WINDOW_S` | `300` | A repeat of the same event type by the same person (stable identity) within this time updates the existing Telegram message instead of sending a new one; after someone presses Seen, a repeat sends a new, notifying reply. `0` disables aggregation. |
| `FIM_ALERT_LATE_AFTER_S` | `60` | An alert delivered later than this after the event is still sent, marked as late. |

### Telegram

| Variable | Default | Description |
|---|---|---|
| `FIM_TELEGRAM_ENABLED` | `false` | Send alerts to Telegram (outbound long polling; no inbound port). |
| `FIM_TELEGRAM_BOT_TOKEN` | *(required when enabled)* | Bot token from BotFather. Never logged. |
| `FIM_TELEGRAM_CHAT_IDS` | *(required when enabled)* | Comma-separated chat ids that receive alerts. The bot answers only these chats. Group messages are not end-to-end encrypted; videos are sent only when someone presses Watch video. |
| `FIM_TELEGRAM_VIDEO_UPLOAD_TIMEOUT_S` | `600` | Time allowed to upload one event clip to Telegram. Uploads take far longer than messages on a home uplink (a 50 MB clip at 1 Mbit/s needs about 7 minutes). |

### Webhook

| Variable | Default | Description |
|---|---|---|
| `FIM_WEBHOOK_URL` | *(none)* | When set, every notified event is POSTed there as JSON (retried until accepted). Treated as a secret. |
| `FIM_WEBHOOK_TIMEOUT_S` | `10` | Timeout of one webhook request. |

### Event log and outbox

| Variable | Default | Description |
|---|---|---|
| `FIM_EVENT_LOG_FILE` | `/events/events.jsonl` | JSON event log: one line per event, near-misses included. Always on. |
| `FIM_OUTBOX_DIR` | `/events/outbox` | Durable queue of pending deliveries, one folder per output; pending alerts survive restarts and outages of the database or the internet. |

### Health

| Variable | Default | Description |
|---|---|---|
| `FIM_HEALTH_CHECK_INTERVAL_S` | `5` | How often camera, disk and delivery health is checked. |
| `FIM_HEALTH_CAMERA_LOST_S` | `30` | Seconds without video (or without a first frame after start-up) before a camera is reported as degraded monitoring; once degraded, any gap longer than `FIM_CAPTURE_STALE_TIMEOUT_S` keeps it degraded. The message arrives within this limit plus `FIM_HEALTH_CHECK_INTERVAL_S`; keep the sum at most 60. |
| `FIM_HEALTH_RESTORE_AFTER_S` | `30` | A degraded camera or system is reported restored only after this long without any problem, so a flapping camera does not flood the chat. |
| `FIM_HEALTH_CAMERA_MAX_RECONNECTS` | `3` | This many reconnections of a camera within `FIM_HEALTH_RECONNECT_WINDOW_S` report it as unstable (short, repeated drops). |
| `FIM_HEALTH_RECONNECT_WINDOW_S` | `600` | Window for counting reconnections. |
| `FIM_HEALTH_MAX_PROCESSING_FAILURES` | `5` | Consecutive frames failing in detection before the camera is reported as degraded. |
| `FIM_HEALTH_DATABASE_BACKLOG_MAX_AGE_S` | `300` | An event waiting longer than this for the database degrades the system. Alerts waiting longer than `FIM_ALERT_LATE_AFTER_S` for Telegram or the webhook do too. |
| `FIM_HEALTH_STATE_FILE` | `/events/health_state.json` | Where the service keeps the degraded states it has announced, so that a restart does not announce them again and still sends "restored" when they clear. In the `events` volume. A missing or damaged file only means starting without that memory. |
| `FIM_HEALTH_STALL_EXIT_S` | `180` | A camera worker with no activity at all (no frame and no stream state), or a blocked event loop, for this long means processing is stuck: the process exits so Docker restarts it. At least 60. |

### Recording and event clips

| Variable | Default | Description |
|---|---|---|
| `FIM_MEDIAMTX_RTSP_URL` | `rtsp://mediamtx:8554` | MediaMTX RTSP relay read by the monitoring service (path `<id>sub`, or `<id>` without a substream). |
| `FIM_MEDIAMTX_PLAYBACK_URL` | `http://mediamtx:9996` | MediaMTX playback server used to export clips. |
| `FIM_RECORDINGS_DIR` | `/recordings` | Recordings folder inside the app container (the `recordings` volume). |
| `FIM_RECORDINGS_MAX_AGE_H` | `168` | Hours a continuous recording segment is kept (MediaMTX deletes older ones). |
| `FIM_RECORDINGS_MAX_GB` | `200` | Disk cap for continuous recordings; `fim recordings retention` deletes the oldest segments above it. Size it to the server's disk. |
| `FIM_RECORDINGS_DISK_MIN_FREE_GB` | `10` | Free space kept on the recordings disk by deleting the oldest recordings; below it after cleanup, disk pressure is reported. |
| `FIM_EVENT_CLIP_DIR` | `/events` | Event clips folder inside the app container (the `events` volume). |
| `FIM_EVENT_CLIP_PRE_S` | `10` | Seconds of video before the event start in an event clip. |
| `FIM_EVENT_CLIP_POST_S` | `20` | Seconds of video after the event start in an event clip. The clip is a fixed window around the start (the fall, the impact, the moment of lying down), so a long event does not make a long clip; it is ready about this long after the start, plus the export. |
| `FIM_EVENT_CLIP_MAX_MB` | `50` | Maximum clip size (the Telegram limit); larger clips are re-encoded at a lower bitrate. |
| `FIM_EVENT_CLIP_RETENTION_DAYS` | `30` | Days an event clip is kept. Event clips are never deleted for disk pressure. |

### Database

| Variable | Default | Description |
|---|---|---|
| `FIM_DB_PASSWORD` | *(required)* | PostgreSQL password, used by both the `postgres` service and the app. Alerts never depend on the database. |
| `FIM_DB_HOST` | `postgres` | PostgreSQL host (the Compose service). |
| `FIM_DB_PORT` | `5432` | PostgreSQL port. |
| `FIM_DB_NAME` | `fim` | Database name. |
| `FIM_DB_USER` | `fim` | Database user. |

### Development

| Variable | Default | Description |
|---|---|---|
| `FIM_SIM_MEDIA_DIR` | `./test_videos` | Development only: host folder with videos for the simulated camera. |
| `FIM_SIM_FILE` | `sample.mp4` | Development only: video file (relative to `FIM_SIM_MEDIA_DIR`) looped by the simulated camera. |
| `FIM_SIM_TARGET` | `rtsp://mediamtx:8554/cam1` | Development only: RTSP URL the simulated camera publishes to. |

## Tools

- `fim probe` — records per-frame timing (camera PTS, arrival time, timestamp quality) for a file or RTSP source and writes CSV, JSON and Markdown summaries under `artifacts/`.
- `fim clip` — exports a time range of a camera's recording as an MP4 file (AAC audio, at most `FIM_EVENT_CLIP_MAX_MB`) with a JSON sidecar holding the requested and actual windows and any gaps.
- `fim recordings retention [--dry-run]` — deletes the oldest recordings above the disk cap or below the free-space floor, and event clips older than their retention; exits 1 on disk pressure. Until the monitoring service runs it continuously, schedule it on the host (for example every 10 minutes with cron: `docker compose run --rm app fim recordings retention`); MediaMTX alone only applies the age limit.
- `fim recordings index` — stores each camera's recorded spans in PostgreSQL and prints the gaps of the last 24 h.
- `fim db upgrade` — applies database migrations.
- `fim alert test` — sends a test message through every enabled output (JSON log, webhook, Telegram) and reports which ones delivered it.
- `fim mediamtx-config --out <file>` — writes MediaMTX's configuration from the camera variables (run by the `mediamtx-config` service).
- `fim analyze` — runs detection, tracking, pose and the detectors on a file or stream; writes `tracks.jsonl`, `events.jsonl` and, with `--overlay`, an annotated video under `artifacts/` (local inspection only). `--zones` applies a zone file.
- `fim zones` — writes a snapshot, safe zones suggested from the beds and sofas it detects, and `zones.png` to check them; `--check` validates a zone file and redraws `zones.png`. `--edit` opens a local page to draw and edit the zones with the mouse. See [`docs/fall-detection.md`](docs/fall-detection.md#zones).
- `fim eval --dataset gmdcsa24|urfd|caucafall --root <folder>` — event-level recall, false alerts per hour and alert latency of the fall rules on a locally downloaded public dataset. See [`docs/evaluation.md`](docs/evaluation.md).
- `fim benchmark detectors` — person-detector recall on fallen and upright people using locally downloaded CAUCAFall videos.
- `fim benchmark pipeline --source <video>` — what one camera costs on this machine: decoding, detection and pose per frame, highest analysis rate, CPU and memory. See [`docs/benchmarks/pipeline.md`](docs/benchmarks/pipeline.md).
- `fim models fetch|verify` — downloads model weights listed in `models.toml` from their original source and verifies their SHA-256.

## License

Apache License 2.0 — see [`LICENSE`](LICENSE), [`NOTICE`](NOTICE) and [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md). Models and datasets are not part of this repository; see [`MODELS.md`](MODELS.md) and [`docs/DATASETS.md`](docs/DATASETS.md).
