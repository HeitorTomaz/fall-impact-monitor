# Fall Impact Monitor — System Architecture

> **Status:** ACCEPTED
> **Version:** 0.2 — 2026-11-08 (supersedes the 0.1 proposal)
> **Repository:** https://github.com/HeitorTomaz/fall-impact-monitor
> **License for original project source:** Apache License 2.0
> **Decisions:** [`docs/adr/`](adr/) · **Contributor conventions:** [`CLAUDE.md`](../CLAUDE.md)

This document describes the current design. Every design change updates it in the same pull request; changes to the stack, services, policies or safety rules also need an ADR.

## 1. Purpose

Fall Impact Monitor is a local, privacy-first **residential** monitoring system for RTSP IP cameras. It is **experimental and assistive** software. It makes otherwise unseen events visible to the people who care for someone at home:

- **possible falls**, including falls followed by immediate recovery;
- **a person lying on the floor**, even when the fall itself was not observed;
- **possible head impacts** (experimental), including impacts without a fall, such as stumbling into a wall and walking on.

Alerts go to optional outputs (a Telegram group first, a generic webhook, a JSON event log). The event video is available on demand. What humans do after an alert is outside the system's scope.

**This is not a medical device.** It does not diagnose, never claims an injury or its absence, and does not replace human supervision.

### 1.1 Goals

1. Analyze video **locally**; raw video never leaves the premises unless a person explicitly requests a clip.
2. Detect falls **without requiring the person to stay on the floor**.
3. Surface possible head impacts with honest, evidence-based wording.
4. Favor **recall**: when in doubt, alert, and make false alarms quick to dismiss.
5. Support several cameras and several people per camera on a CPU-only host.
6. Keep the **detection core** reusable as a library, independent of Telegram and the database.
7. Run as one Docker Compose stack that an installer configures through environment variables only.
8. Keep the public repository redistributable and free of private data.

### 1.2 Non-goals

- Medical diagnosis, clinical-grade accuracy, emergency-service integration or unattended safety guarantees.
- Face recognition, biometric identity or matching to a named person.
- Cloud inference, a mobile app or a web interface (the MVP has none).
- Mandatory GPU or new hardware purchases.
- Monitoring outdoors or outside camera coverage; wearables are a possible future complement.
- Redistribution of private footage, model weights or third-party datasets.

## 2. Safety rules

These rules are non-negotiable. Changing any of them requires an ADR approved by the maintainer.

1. **Recall first.** Missed events are worse than false alarms.
2. **Never suppress by attribution.** No event is suppressed or downgraded because a child is present or because the person's attribution is uncertain. An uncertain person is treated as an adult.
3. **Honest wording.** Say "possible". Never "confirmed", never an injury claim. **A visible collision is not a measured impact**: a single 2D camera cannot reliably tell passing in front of a wall from striking it.
4. **Degraded is not "no incident".** Camera loss, stalled processing, analysis below the minimum rate, disk pressure and notification failures are reported as degraded monitoring.
5. **Never invent time.** No fabricated frames or timestamps. Zero, negative or outlier time deltas are excluded from kinematics. Every evidence records which frames were actually analyzed.
6. **Alerts do not depend on the database.** Delivery survives database outages.
7. **Latency.** The pipeline is designed to alert within 30 s of the event (target) and must be able to operate within 60 s (the design budget used to choose technologies and algorithms). No alert is ever discarded: an alert that exceeds the budget is still sent, marked as late, and the delay is reported as degraded monitoring.
8. **Privacy.** Processing stays local. No face recognition, no biometric identity.

An event is never erased because the person recovered. A person who looks fine after a possible hit may still be injured; the software never labels anyone "cleared".

## 3. Topology

One deployable modular monolith (`app`) plus two off-the-shelf services, all on a private Docker network.

```text
RTSP cameras (N) ── one connection per stream ──> [mediamtx]
                                                    ├─ records main stream (fMP4, 1-min segments, video + audio) → volume recordings/
                                                    ├─ playback API /get (clip by time range)
                                                    └─ internal relay of the substream (video + audio)
                                                             │
[app] ───────────────────────────────────────────────────────┘
  capture (PyAV, per camera) → FrameRecord + timestamp quality + encoded packet buffer (~20 s)
    └→ global scheduler (adaptive cadence, camera priority, batching)
         └→ perception: person detector → OC-SORT → RTMPose  (ONNX Runtime, CPU)
              └→ analysis: fall FSM · person on floor · head-impact fusion ← audio (onset + PANNs)
                   └→ candidates → confirmation (re-decode buffer, post-event observation)
                        └→ events (aggregation, severity, evidence, config snapshot)
                             ├→ sinks (each optional): Telegram · webhook · JSON log   ← independent of DB
                             └→ storage: PostgreSQL (state, history, feedback, segment index)
  recording: segment index, retention, clip export via /get (audio → AAC)
  health: watchdog; camera stale/offline; analysis below minimum; disk → OK / DEGRADED

[postgres]  events, notification states, feedback, health history, configuration snapshots
```

Why a relay: consumer cameras accept only a few concurrent RTSP sessions. MediaMTX holds the single connection per stream, records it and serves the substream to the app, so recording and analysis never compete for camera sessions. **Recording is independent of inference**: the full-rate encoded video is preserved even when the app analyzes fewer frames.

## 4. Modules and the detection-core boundary

Package `fall_impact_monitor` (src layout, Python 3.12):

| Module | Responsibility |
|---|---|
| `api` | Small public facade of the detection core: frames with timestamps (and audio) in, events out |
| `perception` | Detector, tracker, pose, ONNX Runtime/OpenVINO backends, global scheduler with priority and batching |
| `analysis` | Fall state machine, person-on-floor detection, head-impact evidence fusion, audio events, kinematics and filtering, zones, adult/child attribution (logged only) |
| `capture` | PyAV readers (file and RTSP), reconnection with backoff, stall detection, `FrameRecord` and timestamp quality, encoded packet buffer, audio reader |
| `recording` | MediaMTX client, segment index, retention, clip export (remux with AAC audio) |
| `events` | Event model, confirmation, aggregation, severity, near-misses, manual reports |
| `sinks` | Telegram, webhook and JSON log outputs; outbox with retry |
| `health` | Watchdog, per-camera and system states, disk monitoring |
| `storage` | SQLAlchemy models and Alembic migrations |
| `evaluation` | Dataset and model manifests, event-level metrics, reports |
| `config` | `FIM_*` settings, validation, secret masking, configuration hash and snapshot |
| `service` | The monitoring service (`fim run`): per camera, a reader thread (relay → decode) and an analysis thread on the latest frame (perception → detectors → event records), the outputs and Telegram polling, background clip export, periodic maintenance (retention, span index), migrations retried until the database answers, heartbeat for the container health check |
| `cli` | `fim run`, `probe`, `clip`, `zones`, `analyze`, `alert`, `recordings`, `db`, `eval`, `benchmark` |
| `_vendor` | Third-party code copied with its original headers |

### 4.1 Boundary rules

- The **detection core** (`api`, `perception`, `analysis`) does not import `sinks`, `storage`, `recording`, `health`, aiogram or SQLAlchemy. An automated check enforces this.
- Data flows one way: capture → perception → analysis → events → sinks and storage.
- In `pyproject.toml` the core has minimal dependencies; the application (Telegram, database) is an optional extra, so the core can be installed alone.
- Third parties can use the core through the Python `api`, through `fim analyze` (video file → JSON events, also used by evaluation), or by consuming the webhook or JSON log.
- Every detector (fall, person on floor, head impact, audio) and every output can be switched on or off by configuration.

### 4.2 Core contracts

| Contract | Fields |
|---|---|
| `FrameRecord` | camera, seq, pts, time_base, `arrival_monotonic_ns`, `wall_utc`, keyframe, size, timestamp flag (§5.2) |
| `Track` | ephemeral per-camera id, box, score, COCO-17 keypoints (x, y, conf), identity `stable` or `uncertain`, attribution label (logged only) |
| `Evidence` | kind, time window, value, confidence, frames actually analyzed (never claims skipped frames) |
| `Event` | id, camera, category, **confidence score** (0.0–1.0, §9.5), severity, tracks, evidence, window, state, clip reference, config hash, model versions (§10) |

## 5. Capture and time domains

### 5.1 Inputs and analysis stream

- RTSP through the MediaMTX relay, or local video files (MP4, MKV, AVI) for reproducible experiments. The same reader and CLI serve both; only configuration changes.
- PyAV decodes; it exposes per-packet PTS and does not block indefinitely like OpenCV's `read()`. Options: `rtsp_transport=tcp`, `allowed_media_types`, timeouts.
- Reconnection with backoff and stalled-stream detection. The reader records `monotonic_ns` on arrival and RTCP sender-report side data when present.
- **Analysis runs on the substream.** On the reference camera (TP-Link Tapo C113) the substream is 1280×720 at 20 fps according to the vendor's community forum; it is far cheaper to decode than the main stream. The main stream is only recorded.
- Substream and audio are optional per camera. Without a substream, analysis decodes the main stream at reduced resolution. Without audio, head-impact fusion relies on the other evidence.
- The system discovers resolution, codec and frame rate at runtime instead of assuming them.
- **Real time first.** Per camera, a reader thread reads and decodes the analysis stream as it arrives, so `arrival_ns` is the time a frame was received as long as reading and decoding keep up with the camera. The analysis thread takes the most recent decoded frame through a one-slot hand-over (`service/handover.py`): when analysis is slower than the camera, the frames it had no time for are skipped and counted, never queued, so monitoring never falls behind real time and kinematics never see the compressed read times of a draining backlog. Stream states (connected, stale, disconnected) are not dropped to make room for frames. "Full-rate analysis" therefore means as fast as analysis allows.

### 5.2 Time domains and `FrameRecord`

Three time domains are kept separate and never mixed:

1. **Camera PTS** (with its rational time base) — the source's own presentation time.
2. **Monotonic arrival** (`arrival_monotonic_ns`, integer nanoseconds) — includes network and buffering jitter.
3. **UTC wall clock** (`wall_utc`) — for humans, logs and event records only.

Each frame carries a timestamp-quality flag:

| Flag | Meaning |
|---|---|
| `ok` | Plausible positive delta |
| `zero_delta` | Same timestamp as the previous frame |
| `backward` | Timestamp went back (including undetected RTP wrap-around) |
| `outlier_gap` | Forward jump beyond a configured bound |
| `missing` | No timestamp available |

Flagged frames are kept for detection by box and pose but **excluded from kinematics**. Nothing is interpolated or invented; skipped or missing frames are never presented as analyzed.

The pipeline separately tracks nominal fps, decoded fps, frames delivered, frames selected for analysis, analyses completed and their latency, and delta/jitter distributions per time domain.

### 5.3 Encoded packet buffer

Each camera keeps about **20 s** of encoded substream packets (video and audio) in memory, bounded by time. Decoded RGB frames are never buffered: that would cost hundreds of megabytes per camera. The confirmation stage (§9.1) re-decodes the buffer on demand at full rate.

### 5.4 Observed timestamp behavior and the Phase 1 timing protocol

Preliminary local captures showed **irregular source timestamps**: runs of identical timestamps followed by forward jumps. A VLC recording in AVI looks smoother only because AVI stores no per-frame timestamps; it is not evidence of regular capture. Firmware updates may change this behavior, so the camera firmware version is recorded and cameras are not updated automatically.

Phase 1 characterizes timing with `fim probe`:

- `docker compose run --rm app fim probe --source … --duration 60 [--video-only] [--wallclock]`;
- per-packet/frame CSV in `artifacts/`; JSON/Markdown summary with fps per time domain, delta histogram, zero/backward counts, GOP length and frames per second;
- comparisons: direct camera vs MediaMTX relay; main stream vs substream; with and without `allowed_media_types`; wall clock;
- **ground truth**: filming a running stopwatch or blinking LED, in light and dark, at 15/20/25/30 fps;
- a 24–48 h MediaMTX recording run watching for recorder failures (§20).

Anonymized results go to `docs/capture-timing.md`. Installation-specific details stay in the git-ignored `local/` directory. Phase 1 decides which time domain feeds kinematics (§20).

## 6. Recording, retention tiers and clips

### 6.1 Recording

MediaMTX records each camera's **main stream** (video and audio) as **fMP4 segments of 1 minute** in the `recordings` volume. The one-shot `mediamtx-config` service generates MediaMTX's configuration from the same `FIM_*` camera variables on every start, so cameras are declared once (ADR-0013): path `<id>` relays and records the main stream, and path `<id>sub` relays the substream for analysis only.

`fim recordings index` stores the **recorded spans** of each camera in PostgreSQL (`recording_spans`), taken from the playback server's `/list`. MediaMTX merges contiguous segments into one span, so the space between two spans is a **gap** in the recording, for example while the camera was offline. Gaps are listed, never hidden.

### 6.2 Event clips

1. After an event, the app waits until the recording reaches the end of the post-event window (or gives up after a timeout and exports what exists).
2. The window is **20 s before the event start to 60 s after its end** (`FIM_EVENT_CLIP_PRE_S`, `FIM_EVENT_CLIP_POST_S`). It is requested from the MediaMTX playback API `/get` **one recorded span at a time**: the server stops a response at the first gap and refuses a start that falls in a gap. The pieces are joined back to back into one clip; the gaps are cut out and listed. A window too long to fit the size cap at the minimum bitrate (about 19 minutes for 50 MB) is truncated at its end, since the start of an event matters most, and the clip is marked as truncated.
3. Audio (often G711 from cameras) is converted to **AAC** so the MP4 plays everywhere. Video is remuxed, not re-encoded, unless the clip has several pieces or exceeds `FIM_EVENT_CLIP_MAX_MB` (50 MB, the Telegram limit). Audio is padded with silence across a joint so that it stays in sync with the video. In that case it is re-encoded with H.264 at a bitrate computed from the duration, at lower quality, keeping every frame's original timestamp. A clip still above the limit is kept: a large clip is better than none.
4. The clip is stored in the `events` volume as `<camera>/<YYYY>/<MM>/<DD>/<HHMMSS>_<event id>.mp4` (UTC) and kept **30 days** (`FIM_EVENT_CLIP_RETENTION_DAYS`).
5. A JSON sidecar records the **requested and actual** windows and the gaps, so the information survives without the database. The actual window is derived, never assumed:
   - its end is where the request or the recording ends;
   - its start is the first recorded instant, moved earlier by however much the file is longer than the recorded part (the keyframe snap).
6. `fim clip` exports any time range the same way.

### 6.3 Retention tiers

| Tier | Content | Retention | When |
|---|---|---|---|
| Continuous recording | All main-stream segments | By age (`FIM_RECORDINGS_MAX_AGE_H`, MediaMTX) + disk cap (`FIM_RECORDINGS_MAX_GB`) and free-space floor (`FIM_RECORDINGS_DISK_MIN_FREE_GB`), oldest first, never the segment being written (`fim recordings retention`) | MVP |
| Event clips | −20 s/+60 s clips | 30 days; never deleted for disk pressure (pressure is reported instead) | MVP |
| Motion/person archive | Segments with motion or a visible person; segments with neither are discarded | ~7 days, with a disk cap | Post-MVP |
| Protected | Recordings around a manually reported missed event | Until released | With `/missed` |

The archive exists to find **false negatives**. Lower resolution is used only if disk space becomes tight.

### 6.4 Plan B

If MediaMTX's recorder fails the long-run test, recording falls back to `ffmpeg -f segment` with presets adapted from a permissively licensed project (see `THIRD_PARTY_NOTICES.md`). The playback and clip interfaces of `recording` stay the same.

## 7. Perception

| Stage | Choice | Notes |
|---|---|---|
| Person detector | **Selected by benchmark** among YOLOX-Tiny/S, YOLOX-HumanArt, YOLO11/YOLO26, RTMDet, D-FINE | Criterion: recall for distant, lying and unusually posed people within the CPU budget, measured on the target host class and on evaluation data. YOLO11/YOLO26 are used only as exported ONNX; the `ultralytics` package is never a dependency of the repository or the image |
| Tracker | **OC-SORT** (Roboflow `trackers`), re-identification off | Accepts absolute timestamps, which helps with irregular source timing |
| Pose | **RTMPose-s/m** (most accurate body variant), top-down on per-person crops | Pre/post-processing copied from rtmlib instead of depending on it. A crop of a lying person is rotated 90° before pose |
| Whole-body pose | **RTMW** (post-MVP) | Only in the confirmation stage, for hand-to-head evidence. In the MVP, confirmation uses RTMPose |
| Runtime | **ONNX Runtime CPU**; OpenVINO optional (requires AVX2) | **CUDA only for NVIDIA GPUs with compute capability ≥ 7.5**; older GPUs are not supported by current CUDA/cuDNN builds |

MediaPipe was dropped (multi-person, distance and lying-person limitations). ViTPose and similar models are too heavy for CPU. Smaller or quantized models are adopted only when the benchmark justifies them.

## 8. Multi-camera scheduling

Several cameras are a requirement. A **global scheduler** shares one inference queue across cameras.

### 8.1 Safe accelerations (MVP)

- Analysis on the substream.
- Empty scene: only cheap motion detection.
- Person present: the detector runs at a few fps; the tracker covers frames in between.
- Pose per person on demand, faster during sudden movement or posture change.
- One inference queue batching crops from all cameras.
- OpenVINO or smaller/quantized models when the benchmark justifies them.

### 8.2 Priority and minimum rate

Priority across cameras: **ongoing event > person on floor > person moving > person still > empty scene**.

Each camera has a configurable **minimum analysis rate**. Below it, the camera enters **degraded monitoring** and the outputs are told. A camera is never silently starved.

Capacity is documented by benchmark as "N cameras at X fps on hardware class Y" (`fim benchmark`).

### 8.3 Later, after validation

Reduced (never zero) analysis rate for a *confirmed* child under CPU pressure, returning to full rate at the first sign of a fall or uncertain identity. This waits for the attribution validation in §9.7.

## 9. Detection logic

### 9.1 Candidates and confirmation

1. **Real time.** The substream is decoded, the scheduler picks what to analyze, and perception plus detectors produce **candidates**.
2. **Confirmation.** Every candidate (fall or impact) is confirmed before alerting:
   - the packet buffer is re-decoded at full rate over the event window;
   - pose runs on every frame with a valid timestamp (RTMPose in the MVP; RTMW is added post-MVP for hand-to-head on impact candidates);
   - the system watches a few seconds after the event: did the person get up, stay down, stagger, or raise a hand to the head?
3. **Alert** within **30 s** of the event (target), within the **60 s** design budget; late alerts are still sent, marked late, and reported as degraded.

Confirmation improves precision without requiring the person to stay down: a fall with immediate recovery still alerts. If full-rate re-analysis is cut from a release (§18.3), confirmation uses the real-time results plus post-event observation.

### 9.2 Fall: two-level state machine

Per person: `normal → suspected → fallen → recovering → normal`, adapted from an Apache-2.0 implementation (see `THIRD_PARTY_NOTICES.md`).

- Descent is measured by vertical velocity of hips and shoulders normalized by body height, trunk angle and box aspect-ratio change.
- **Level 1 — `possible_fall`**: alerted after confirmation, even if the person recovers.
- **Level 2 — `person_down`**: a follow-up alert, "person on the floor for X s", if the person stays down.
- `suspected → normal` without a fall becomes a **near-fall** near-miss (§9.5).
- A possible fall that ends inside a **safe zone** (lying down in bed) is a near-miss; a lying person who leaves such a zone and lies outside every zone (out of bed onto the floor) arms a suspicion (ADR-0012, ADR-0014).

### 9.3 Person on the floor without an observed fall

Slow slides, hidden falls and a person entering the view already down are covered separately. A person **lying outside safe zones** (sofa, bed, a mattress on the floor, play mat) for longer than X s raises `person_down` with "Person lying on the floor for X s (fall not observed)". X is configurable and calibrated on site. **Children are not exempt.**

Zones are set at installation with `fim zones`, which suggests safe zones from the beds and sofas it detects in a snapshot; the installer checks and edits them. There is one kind, the **safe zone**: a soft place (bed, sofa, a mattress on the floor, play mat). It applies to the fall, the person-on-the-floor and the head-impact rules; everything outside safe zones is treated as a place where a head can hit something (ADR-0014). Zone files are git-ignored.

### 9.4 Possible head impact (experimental)

Evidence within a window of a few seconds:

- **(a) kinematic**: abrupt deceleration or reversal of the head anywhere outside safe zones;
- **(b) audio**: a thud (cheap onset detector as a gate, then PANNs Cnn14_8k);
- **(c) reaction**: staggering, hand to head, sudden sitting, or a fall.

An alert needs **≥ 2 independent evidences** (configurable minimum) **and** a confidence score at or above the alert threshold (§9.5). One evidence, or a score below the threshold, becomes a near-miss. A fall with a fast head stop is therefore usually also a possible head impact. No single kinematic detector can measure an impact at typical camera frame rates, which is why evidence is fused and the label stays experimental. Implementation and formula: [`head-impact.md`](head-impact.md). Instead of the full-rate confirmation stage (not implemented yet), the pipeline analyses every frame while a head outside safe zones moves fast.

**Kinematics** use only frames flagged `ok`, with keypoints smoothed by a One Euro filter and causal derivatives. With poor timing, kinematic evidence loses confidence but box- and pose-based alerts continue.

### 9.5 Confidence score, alert threshold and near-misses

- The detection core returns a **confidence score for every event**: 0–100 % to people, **0.0–1.0 in the API**.
- An event alerts when its score is at or above **`FIM_ALERT_MIN_CONFIDENCE`** (default **0.5**, i.e. 50 %). For `possible_head_impact` the ≥ 2 independent evidences gate (§9.4) applies in addition.
- Below the threshold, the event is a **near-miss**: stored with its evidence and score, and **never notified**. Near-falls (`suspected → normal`) and single-evidence impacts are near-misses as well. Near-misses support calibration and later review.
- **Honesty about the score:** scores from rule-based detectors are **not calibrated probabilities**. "0.7" does not mean a 70 % chance of a real fall until the scores have been calibrated on site data; calibration maps scores toward probabilities. Until then the threshold is a tuning setting, chosen with recall first.
- Alerts show the score (§9.9).

### 9.6 Audio

Audio is **on by default** (recording plus the experimental thud detector) and can be disabled per installation (`FIM_AUDIO_ENABLED`). The service reads each camera's audio from the relay's main path, resamples it to 8 kHz mono and runs an energy-onset detector in memory; nothing from this detector is stored or sent. The PANNs classifier is not implemented yet (issue #6 cut order).

### 9.7 Children and attribution

- Adult/child/uncertain attribution runs **for logging only**. It never suppresses, downgrades or reroutes an alert.
- An uncertain person is treated as an adult; an adult hurt near a child still alerts.
- **Opt-in child handling (post-MVP).** `FIM_CHILD_EVENTS` selects `alert` (default), `silent` or `ignore` for events attributed to a child with at least `FIM_CHILD_ATTRIBUTION_MIN_CONFIDENCE`. Uncertain people, identity doubts and events involving an adult always alert; ignored events are persisted with the reason; tracking and analysis never stop. Enabling `ignore` accepts the risk that a mislabelled adult is not notified (documented in the README). See ADR-0010.
- Reduced analysis cadence (never zero) for *confirmed* children waits for a validation on site data showing that adult recall does not drop.
- Inferring age may constitute biometric categorization under EU law; review before enabling any behavior based on it or distributing it.

### 9.8 Aggregation and the repeat rule

- Same person, **stable** identity, same event type, within a few minutes: the existing message is **updated**.
- A different person or an **uncertain** identity always produces a **new** alert.
- While nobody has pressed **Seen**, repeats only update the message. After **Seen**, a repeat produces a new reply that notifies the group, so a second fall is never silent.

### 9.9 Message wording

"Possible fall", "Person on the floor for X s", "Person lying on the floor for X s (fall not observed)" or "Possible head impact (experimental)", with time, camera, evidence and the **confidence score** as a percentage. Never "confirmed". A short footer states that the system does not replace supervision.

## 10. Events and storage

### 10.1 Categories and states

| Category | Meaning |
|---|---|
| `possible_fall` | Fall candidate that passed the confirmation stage, with or without recovery |
| `person_down` | Person on the floor after an observed fall, or lying outside safe zones without an observed fall |
| `possible_head_impact` | Experimental; ≥ 2 independent evidences and score ≥ threshold |
| `manual_report` | A missed event reported by a person (`/missed`) |
| `monitoring_degraded` | A camera or the system entered a degraded state |
| `monitoring_restored` | The degraded condition cleared |

States: `detected → notified → seen → feedback` (feedback is `false_alarm` or `confirmed`, given by a person). Near-misses (score below `FIM_ALERT_MIN_CONFIDENCE`, §9.5) keep their category, are stored with a near-miss marker and stay `detected`.

### 10.2 PostgreSQL

PostgreSQL (SQLAlchemy 2, Alembic, asyncpg) stores:

- events, evidence summaries and per-segment summaries (**nothing per frame**);
- notification states and attempts;
- human feedback and manual reports;
- the recording-segment index;
- health history;
- a **hash and snapshot of the active configuration** and model versions with every event.

It **never stores video, images or other blobs**. Timestamps are `timestamptz` in UTC. Configuration lives in environment variables, not in the database.

### 10.3 Alert path independent of the database

The sinks receive each event **before it is persisted**; the alert never waits for the database. Persistence happens afterwards and retries until the database accepts the write. Pending alerts survive a restart even when the database and the internet fail together. Startup ordering alone is not relied on: the app tolerates the database disappearing at any time.

Mechanism (`sinks/`):

- An event becomes an **event record**: the detector event plus its UTC times and the hash of the configuration that produced it. A snapshot of that configuration, with secrets masked, is stored once per hash.
- `Dispatcher.submit` writes one file per accepting output into a **durable outbox** (`FIM_OUTBOX_DIR`, one folder per output) and returns at once; it is safe to call from capture threads.
- One asyncio worker per output delivers its queue in order. A failing item is retried with exponential backoff (1 s up to 60 s), or after the delay Telegram asks for; it is deleted only once delivered. A slow or unreachable output never delays the others.
- **The database is one of these outputs.** It receives every event (near-misses included), each delivery result (`notifications`) and each human answer (`feedback`) through its own queue, so it can be down for hours without losing anything or delaying an alert.
- **Telegram has one output per configured chat**, so a retry never sends twice to a chat that already got the alert.

## 11. Outputs

All outputs consume the same event stream; each is optional.

| Output | Enabled by | Notes |
|---|---|---|
| Telegram | `FIM_TELEGRAM_ENABLED` | aiogram 3, outbound long polling, answers only configured chats |
| Webhook | `FIM_WEBHOOK_URL` set | POST JSON per event |
| JSON event log | Always on | Local file, one event per line |

### 11.1 Telegram workflow

- An alert is **text**, with buttons **Seen / False alarm / Confirmed**. These buttons record human feedback; the system's own wording never says "confirmed".
- **View video** sends the event clip to the group as a normal message (forwarding not restricted, not deleted afterwards).
- `/status` reports per-camera and system health on demand.
- `/missed` (pt-BR alias `/perdido`) reports a missed event: it creates a `manual_report` and protects the recordings around that time.
- Degraded and restored messages (§12).

Aggregation state (which message to update, whether someone pressed Seen) lives in memory, per chat: after a restart, the next event starts a new message. Repeats are keyed by camera, category and track id, and only for a stable identity.

Telegram limits shape the design: bots upload files up to **50 MB** (clips are sized to fit), and should send about **one message per second per chat** (fewer in groups). The outbox queues and retries; aggregation keeps volume low. Delivery to Telegram does not prove that anyone saw the alert.

### 11.2 Localization

User-facing notification texts are localized, **English by default** and **pt-BR** available, selected by an environment variable. Everything else (logs, CLI output, JSON fields) is English.

## 12. Health and degraded monitoring

- The **watchdog** tracks per-camera states (stream stale or offline, analysis below the minimum rate) and system states (disk usage, stalled work, notification failures).
- Each camera and the system are **OK** or **DEGRADED**. Transitions emit `monitoring_degraded` and `monitoring_restored`. A lost camera is reported within 60 s (configurable).
- If work stalls, the watchdog exits the process so Docker restarts it.
- Implementation (`health/`, `service/health.py`): every `FIM_HEALTH_CHECK_INTERVAL_S` the service checks each camera (no video for `FIM_HEALTH_CAMERA_LOST_S`, including never connected; analysis below `FIM_ANALYSIS_MIN_FPS` while video flows), the disk (`FIM_RECORDINGS_DISK_MIN_FREE_GB`) and the outbox (an alert or a database write waiting longer than `FIM_HEALTH_BACKLOG_MAX_AGE_S`). A component is restored only after `FIM_HEALTH_RESTORE_AFTER_S` without any problem, so a flapping camera gives one degraded and one restored message. Degraded and restored messages are events like the alerts: they go through the outbox to every output, and the `events` table is the health history. The watchdog exits with code 70 when a camera worker shows no activity (no frame and no stream state) for `FIM_HEALTH_STALL_EXIT_S`; an offline camera keeps reporting stream states and does not trigger it.
- Without internet, detection continues locally and alerts queue; they reach Telegram when connectivity returns.

**Limitation:** if the **whole host** fails (power, hardware, OS), nobody is notified automatically. There is no external heartbeat in the MVP. `/status` is the manual check: a bot that does not answer means the host is down. An optional external heartbeat is planned post-MVP.

## 13. Configuration

- Every main setting is a `FIM_*` environment variable read by pydantic-settings, validated at start-up (fail fast) and documented in the README variable table and `.env.example`.
- Names follow `FIM_<PROCESS>_<WHAT>_<UNIT>` (see `CLAUDE.md` §11): one prefix per process (`FIM_CAPTURE_*`, `FIM_PERSON_DETECTOR_*`, `FIM_TRACKER_*`, `FIM_ANALYSIS_*`, `FIM_PERSON_DOWN_DETECTOR_*`, `FIM_ALERT_*`, `FIM_RECORDINGS_*`, `FIM_EVENT_CLIP_*`, `FIM_DB_*`) and the unit last.
- **Cameras** are declared through variables: N cameras, each with its main stream, optional substream, optional audio and per-camera minimum analysis rate. Adding a camera needs configuration only.
- Each detector and output has its own switch.
- The alert threshold is `FIM_ALERT_MIN_CONFIDENCE` (0.0–1.0, default 0.5).
- Other thresholds and timings (X seconds on the floor, minimum evidences, aggregation window, degraded timeout, retention) are named settings; detector code has no magic numbers.
- **Zone files** (safe zones) are referenced by variables and stored in git-ignored paths.
- RTSP URLs and tokens are masked in logs. A hash and snapshot of the configuration are stored with every event.

The README is the authoritative list of variable names.

## 14. Deployment

### 14.1 Services

| Service | Image | Volumes | Published ports |
|---|---|---|---|
| `mediamtx-config` | The app image, one-shot | `mediamtx-config` | None |
| `mediamtx` | Official image, pinned version | `recordings`, `mediamtx-config` (read-only) | None (LAN RTSP only if a variable enables it) |
| `postgres` | Official image, pinned major version | `pgdata` | None |
| `app` | Built locally from the `Dockerfile` (non-root user); runs `fim run`; health check on a heartbeat file rewritten every 10 s | `recordings`, `events`, `artifacts` | None |

Every long-running service has `restart: unless-stopped` and a health check. The one-shot services (`init-volumes`, `mediamtx-config`) run to completion before the services that need them. Images are **built locally and never published**. Kubernetes may come later.

### 14.2 Models at build time

Weights are downloaded **during the image build** from their original sources, listed in a manifest (source URL, SHA-256, license, provenance), and **verified against the SHA-256**. Nothing is downloaded at runtime, so the system runs **offline**. Where a model needs conversion (e.g., PANNs to ONNX), it happens in a separate build stage so heavy tooling does not reach the runtime image.

### 14.3 Updates and rollback

- `git pull && docker compose up -d --build`; Alembic migrations run at app start-up; volumes preserve data.
- The previous app image is kept under a **rollback tag**.
- [`deployment.md`](deployment.md) holds the installation guide, the update and rollback procedures and the on-site update checklist: services healthy, `/status` answering, test alert received.

### 14.4 Backup and restore

A manual, documented procedure with `docker compose exec postgres pg_dump` / `pg_restore`, tested at least once and again after schema changes. Backup automation is post-MVP.

### 14.5 Development

Development runs on Docker Desktop (Windows) or Linux. Tests and CLI commands run through Compose (`docker compose run --rm app …`); native `uv run` is optional. Production targets a Linux server with Docker.

## 15. Privacy and security

- Processing is local. Alerts are text. Video reaches Telegram **only on demand** and then stays on Telegram's servers; group chats are not end-to-end encrypted. This is documented for users.
- Third-party telemetry is off: official ONNX Runtime builds send usage events to Microsoft by default, so every image sets `ORT_DISABLE_TELEMETRY=1` and importing the package sets it when it is not set.
- Audio is on by default. Installers document the occupants' consent; audio can be disabled.
- No faces, no biometrics. Age attribution is logged only (§9.7).
- Secrets live only in environment variables or the git-ignored `.env`; `.env.example` holds placeholders.
- No ports are published to the internet. Telegram uses outbound long polling. PostgreSQL and MediaMTX stay on the internal network.
- Containers run as non-root. Downloaded weights, datasets and archives are untrusted: SHA-256 verified, never unpickled.
- Network recommendation: block the cameras' internet access where possible but allow NTP, since some cameras misbehave when fully cut off.
- The repository never contains recordings, event clips, `local/`, `test_videos/`, `artifacts/`, weights, datasets, credentials, household details or images of identifiable people. CI uses synthetic or redistributable fixtures only.

## 16. Models, datasets and licensing policy

**Quality first for installations.** Installations choose models and datasets by benchmark on evaluation data, not by license class. Any published, openly downloadable model or dataset may be used privately by the installer; its terms are recorded, it is never redistributed, and its authors are cited. Datasets behind registration or request forms are out of scope. Non-commercial datasets are used for private evaluation and training only.

**The repository stays redistributable** under Apache-2.0:

- no GPL, AGPL or unlicensed code is copied into it;
- no weights or datasets are committed; installers download them, with documented terms;
- weights trained on household recordings or on non-commercial or research-only data are never published;
- commercial adopters replace restricted models and data with permissive alternatives.

Notable cases: YOLO11/YOLO26 weights are AGPL-3.0 (used only as exported ONNX); RTMPose/RTMW weights have no separate license and were trained on data with mixed terms; PANNs weights are CC BY 4.0 (attribution and ONNX conversion noted); PyAV wheels bundle FFmpeg builds with GPL components, recorded because no images are distributed.

Evaluation data starts with CAUCAFall and GMDCSA24 (redistributable with attribution) plus openly downloadable non-commercial fall datasets used privately.

Where this is recorded:

| File | Content |
|---|---|
| [`MODELS.md`](../MODELS.md) | Each weight: source, license, training data, SHA-256, conversions |
| [`docs/DATASETS.md`](DATASETS.md) | Each dataset: source, license, permitted use (private / redistributable), citation |
| [`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md) | Each dependency and vendored file: URL, version/commit, license, copyright, changes |
| [`CREDITS.md`](../CREDITS.md) | Projects used only as ideas, papers and datasets |
| `NOTICE`, `LICENSES/` | Required notices; full license texts by SPDX identifier |

Vendored code lives in `src/fall_impact_monitor/_vendor/<origin>/` with original headers, SPDX identifier and an `Adapted from <url>@<commit>; changed: …` line.

## 17. Evaluation and acceptance

### 17.1 Metrics

Measurements are **event-level**, not per frame: events are matched with a time tolerance (mir_eval `match_events`), and each run reports recall, false alerts per hour and alert latency per dataset or scenario. Results are measurements, never guarantees or clinical claims.

### 17.2 Staged tests

1. **Unit and scenario tests** on synthetic frames, timestamps, keypoint tracks and audio (timing edge cases, falls with immediate recovery, slow slides, lying in safe zones, children playing, occlusion, identity swaps). These run in CI.
2. **Dataset evaluation** locally with `fim eval`.
3. **Staged scenes** by an adult volunteer onto a mattress in a test environment. No demonstrations with older adults or children.
4. **Fault injection**: camera, database, disk, internet.
5. **On-site calibration**: during the first period, alerts go only to the installer, who tunes thresholds with on-site data (including near-misses and feedback) before other recipients join.

### 17.3 MVP acceptance criteria

| # | Criterion | Blocks install |
|---|---|---|
| 1 | Clean install: on a Linux server with Docker, editing only `.env` and running `docker compose up -d` brings all three services to healthy; the system recovers after reboot | Yes |
| 2 | Degraded monitoring: an unplugged camera is reported within 60 s (configurable), restoration is reported, reconnection is automatic in 3 of 3 trials | Yes |
| 3 | Staged falls: ≥ 20 attempts (≥ 10 with immediate recovery, varied positions and directions); ≥ 16 alerted; latency p95 ≤ 30 s, none above 60 s | Reported |
| 4 | Non-fall activities (sitting quickly, lying on a sofa, bending, picking up objects): false alarms counted per hour | Reported |
| 5 | Video: in 100 % of alerts, View video delivers a playable MP4 ≤ 50 MB with requested and actual windows stated | Yes |
| 6 | Database stopped: alerts still reach Telegram | Yes |
| 7 | Dataset evaluation: recall and false alerts per event recorded (no target) | Reported |
| 8 | Phase 1 timing report: main vs substream, direct vs MediaMTX, ground truth, 24–48 h run | Reported |
| 9 | Repository: CI green (ruff, mypy, pytest on synthetic fixtures); no secrets or private media in Git | Yes |
| 10 | Two simultaneous cameras (one real, one looped file) without degraded state | Reported |
| 11 | If head impact ships: ≥ 10 staged soft contacts (head touching a pillow on a wall, with a thud) and ≥ 10 near-contacts, reported as experimental | Reported |

At least one staged fall must also produce an alert end to end before installation.

## 18. Roadmap

Delivery is milestone-driven: **MVP install** (first site installation, **2026-11-30**) and **Post-MVP**. Development and deployments continue after the first installation.

### 18.1 MVP core (required to install)

1. Compose stack (`mediamtx`, `postgres`, `app`) with local builds, `.env.example`, README variable table, update procedure, tested backup/restore.
2. Ingestion via MediaMTX: reconnection, timestamp flags, packet buffer; main-stream recording with age + disk-cap retention.
3. CPU perception: benchmark-selected detector → OC-SORT → RTMPose, adaptive cadence, multi-camera scheduler.
4. Two-level fall detection and person-on-floor without an observed fall (safe zones, zone tool), with aggregation and near-misses.
5. Events in PostgreSQL with a database-independent alert path.
6. Telegram: alerts, buttons, View video, `/status`, degraded/restored.
7. Health: camera stale/offline, analysis below minimum, disk.
8. Modularity: detection core as a library, `fim analyze` for video files, optional outputs (Telegram, webhook, JSON log), detectors switchable by configuration.

### 18.2 Protected item

**Possible head impact** (fusion, re-analysis, audio onset and PANNs) ships at installation if ready; otherwise it is the first item of the next deployment.

### 18.3 Cut order

If time runs short, the webhook goes first (it does not serve the first installation). Then, in order:

1. motion/person retention;
2. `/missed`;
3. child-attribution logging;
4. the PANNs audio classifier;
5. full-rate frame-by-frame re-analysis;
6. head impact itself, last.

### 18.4 Post-MVP increments (proposed order)

1. Head impact (if not shipped) and on-site calibration adjustments.
2. Small temporal classifiers trained locally, added to the rules to reduce false alarms.
3. Motion/person retention, `/missed`, near-miss review.
4. Reinforced head impact: depth estimation and RTMW on candidates; rule-based staggering and hand-to-head, with a trained classifier if enough open data exists.
5. More cameras per installation, with measured capacity.
6. Validation of adult/child attribution; then silent routing and reduced rate for confirmed children; evaluation of ephemeral appearance re-identification.
7. Experiments: candidate review by a local vision-language model (annotates only, never suppresses); optional NVIDIA GPU acceleration (compute capability ≥ 7.5).
8. Community integrations: MQTT with Home Assistant discovery; detection core published on PyPI.
9. Optional external heartbeat, backup automation, web interface, Helm/Kubernetes, ARM64.

Out of scope for now: cross-camera event correlation, a remote monitoring endpoint, detailed metrics in the database.

## 19. Known limitations

- Not a medical device; it does not diagnose and does not replace supervision.
- A single camera cannot confirm head contact; "possible head impact" is experimental.
- Slow falls, occlusions, distant people and events outside the field of view may be missed.
- Kinematic evidence depends on source timestamps, which can be irregular (§5.4).
- If the whole host fails, nobody is notified automatically.
- Without internet, alerts queue and do not reach Telegram until it returns.
- Videos sent to the group remain on Telegram's servers.
- Adult/child attribution does not drive alerts.
- Thresholds need on-site calibration; dataset results do not transfer directly to a home.
- Confidence scores from rule-based detectors are not calibrated probabilities until they are calibrated on site data.

## 20. Open questions

1. **Timestamp domain for kinematics.** Camera PTS, monotonic arrival or a combination? Decided by the Phase 1 ground-truth tests.
2. **Detector choice.** Which candidate gives the best recall for lying and distant people within the CPU budget? Decided by the perception benchmark.
3. **MediaMTX recorder robustness.** Does the recorder survive timestamps that go backwards (an upstream fix was still open at the time of writing)? Decided by the 24–48 h run; plan B in §6.4.
4. **Capacity.** How many cameras per hardware class at the minimum analysis rate?
5. **Outbox mechanism** for alerts that must survive simultaneous database and internet failure.
6. **Remote administration** of an installation (out of scope for the MVP).

## 21. References

Technical components:

- MediaMTX: https://github.com/bluenviron/mediamtx
- PyAV: https://github.com/PyAV-Org/PyAV
- ONNX Runtime: https://onnxruntime.ai/ · OpenVINO: https://github.com/openvinotoolkit/openvino
- YOLOX: https://github.com/Megvii-BaseDetection/YOLOX
- RTMPose / RTMW (MMPose): https://github.com/open-mmlab/mmpose · rtmlib: https://github.com/Tau-J/rtmlib
- Roboflow `trackers` (OC-SORT): https://github.com/roboflow/trackers
- PANNs: https://github.com/qiuqiangkong/audioset_tagging_cnn
- aiogram: https://github.com/aiogram/aiogram
- Telegram Bot API: https://core.telegram.org/bots/api · limits: https://core.telegram.org/bots/faq
- mir_eval: https://github.com/mir-evaluation/mir_eval
- 1€ filter: https://gery.casiez.net/1euro/

Camera:

- TP-Link RTSP/ONVIF integration: https://www.tp-link.com/us/support/faq/2680/
- TP-Link RTSP FAQ: https://www.tp-link.com/us/support/faq/4465/

Papers:

- Cao et al., "Observation-Centric SORT", arXiv:2203.14360.
- Jiang et al., "RTMPose: Real-Time Multi-Person Pose Estimation based on MMPose", arXiv:2303.07399.
- Kong et al., "PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition", arXiv:1912.10211.

Project:

- Issues and milestones: https://github.com/HeitorTomaz/fall-impact-monitor/issues
- Full attribution: `THIRD_PARTY_NOTICES.md`, `MODELS.md`, `docs/DATASETS.md`, `CREDITS.md`.
