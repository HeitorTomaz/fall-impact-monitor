# CLAUDE.md — Fall Impact Monitor

Durable conventions for everyone who changes this repository, human or coding agent.
Current design: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md). Decisions: [`docs/adr/`](docs/adr/).

## 1. What this project is

Fall Impact Monitor is a local, privacy-first residential monitoring system for RTSP IP cameras. It detects:

- **possible falls**, including when the person gets up immediately;
- **people lying on the floor**, even when the fall itself was not observed;
- **possible head impacts** (experimental), including impacts without a fall.

Events go to optional outputs (Telegram first, a generic webhook, a JSON event log). Evidence clips stay on local disk. The detection core is a reusable library.

This is experimental assistive software. It is not a medical device, never diagnoses injury and does not replace human supervision.

## 2. Safety rules (non-negotiable)

Changing any of these requires an ADR approved by the maintainer.

1. **Recall first.** When in doubt, alert. Missed events are worse than false alarms.
2. **No suppression by attribution unless the installer opts in.** By default, no event is suppressed or downgraded because a child is present or attribution is uncertain. An uncertain person is always treated as an adult. The opt-in child modes (`silent`, `ignore`; see ADR-0010) apply only to confidently attributed children, never when an adult or an identity doubt is involved, and ignored events are still persisted.
3. **Honest wording.** Say "possible". Never "confirmed", never an injury claim. A visible collision is not a measured impact.
4. **Degraded is not "no incident".** Camera loss, stalled processing, analysis below the minimum rate, disk pressure and notification failures are reported as degraded monitoring.
5. **Never invent time.** Do not fabricate frames or timestamps. Exclude zero, negative or outlier time deltas from kinematics. Record which frames were actually analysed.
6. **Alerts do not depend on the database.** Delivery must survive database outages.
7. **Latency.** The pipeline is designed to alert within 30 s of the event (target) and must be able to operate within 60 s (the design budget used to choose technologies and algorithms). No alert is ever discarded: an alert that exceeds the budget is still sent, marked as late, and the delay is reported as degraded monitoring.
8. **Confidence threshold.** Every event carries a confidence score (0.0–1.0, shown as a percentage). Only events at or above the configured threshold (default 0.5) are notified; the rest are persisted as near-misses. Rule-based scores are not calibrated probabilities until calibrated on site data.
9. **Privacy.** Processing stays local. No face recognition and no biometric identity. Video leaves the premises only when a person explicitly requests it.

## 3. Language

- Conversation with the maintainer: Brazilian Portuguese (pt-BR).
- Everything in the repository is English: code, identifiers, comments, docstrings, tests, docs, config, issues, PRs, branches, commit messages, ADRs, logs and CLI output.
- Exception: user-facing notification texts are localized (English default, pt-BR available) and selected by an environment variable.
- Do not reference unrelated projects, organizations or products. Do not commit one-off prompts or chat transcripts.

## 4. Privacy and repository hygiene

Never commit, publish, upload, or paste into issues, PRs or logs:

- `local/`, `test_videos/`, recordings, event clips, `artifacts/`, model weights, datasets;
- `.env` files, credentials, RTSP URLs, IP addresses, Telegram tokens or chat IDs;
- household details (names, rooms, routines, health information), the hardware of any private installation, or measurements taken from private recordings;
- images or video of identifiable people, especially children.

Private notes and measurements belong in the git-ignored `local/` directory. Public documents name the reference camera model and describe anonymized, reproducible methods only. Before each commit, run `git status --ignored` and review the diff for secrets. Never send private media to third-party services, including for debugging.

## 5. Approved stack

| Area | Choice |
|---|---|
| Language and tooling | Python 3.12, uv (lockfile committed), ruff (lint + format), mypy, pytest |
| RTSP relay and recording | MediaMTX (official image) |
| Capture and decoding | PyAV |
| Person detection | Selected by benchmark: YOLOX, YOLO11/YOLO26 (exported ONNX only), RTMDet, D-FINE |
| Tracking | OC-SORT from Roboflow `trackers`, re-identification disabled |
| Pose | RTMPose, top-down on per-person crops; RTMW whole-body later (post-MVP), confirmation stage only |
| Inference runtime | ONNX Runtime (CPU); OpenVINO optional; CUDA only for compute capability ≥ 7.5 |
| Audio | Onset detector + PANNs Cnn14_8k |
| Database | PostgreSQL, SQLAlchemy 2, Alembic, asyncpg |
| Outputs | Telegram via aiogram 3 (long polling), generic webhook, JSON event log |
| Configuration | Environment variables (`FIM_*`) via pydantic-settings |
| Deployment | Docker Compose (`mediamtx`, `postgres`, `app`), local builds only |
| CI | GitHub Actions |

Adding a service, adding a dependency under a new license class, or replacing any row above requires an ADR.

## 6. Architecture and module boundaries

One deployable, modular monolith. Package `fall_impact_monitor` (src layout):

| Module | Responsibility |
|---|---|
| `api` | Small public facade of the detection core: frames with timestamps (and audio) in, events out |
| `perception` | Detector, tracker, pose, inference backends, global scheduler |
| `analysis` | Fall state machine, person-on-floor detection, head-impact evidence fusion, audio events, kinematics and filtering, zones, adult/child attribution (logged only) |
| `capture` | PyAV readers (file and RTSP), reconnection, `FrameRecord` and timestamp quality, encoded packet buffer, audio reader |
| `recording` | MediaMTX client, segment index, retention, clip export |
| `events` | Event model, confirmation, aggregation, severity, near-misses, manual reports |
| `sinks` | Telegram, webhook and JSON log outputs; outbox with retry |
| `health` | Watchdog, per-camera and system states, disk monitoring |
| `storage` | SQLAlchemy models and Alembic migrations |
| `evaluation` | Dataset and model manifests, event-level metrics, reports |
| `config` | Settings, validation, secret masking, configuration snapshot |
| `cli` | `fim` entry points: `probe`, `clip`, `zones`, `analyze`, `eval`, `benchmark` |
| `_vendor` | Third-party code copied with its original headers |

Boundary rules:

- The detection core (`api`, `perception`, `analysis`) must not import `sinks`, `storage`, `recording`, `health`, aiogram or SQLAlchemy. An automated check enforces this.
- Data flows one way: capture → perception → analysis → events → sinks and storage.
- Every detector and every output can be switched on or off by configuration.

## 7. Coding standards

- Type hints everywhere; mypy passes (strict for the detection core).
- `ruff format` and `ruff check` are clean. No dead code, no speculative abstractions.
- Prefer small pure functions in `analysis` and `perception`; keep I/O at the edges.
- Put units in names (`_s`, `_ms`, `_ns`, `_px`). Monotonic time is integer nanoseconds. Never mix time domains (camera PTS, monotonic arrival, UTC wall clock).
- Use bounded queues only; nothing may block the alert path.
- Fail fast on invalid configuration at start-up. At runtime, degrade and report instead of failing silently; the watchdog exits the process when work stalls so Docker restarts it.
- Logs are structured JSON in English and never contain secrets (mask RTSP URLs and tokens).
- Thresholds and timings are named, documented settings. No magic numbers in detector code.
- Copied code lives in `_vendor/<origin>/` with its original headers, an SPDX identifier and an `Adapted from <url>@<commit>; changed: …` line, and is listed in `THIRD_PARTY_NOTICES.md`.

## 8. Testing, evaluation and performance

- Write the test first for new behaviour; reproduce bugs with a failing test before fixing them.
- Unit tests use synthetic frames, timestamps, keypoint tracks and audio. Cover timing edge cases: zero deltas, backward steps, forward jumps, RTP wrap-around, missing timestamps.
- Detector scenario tests cover falls with immediate recovery, slow slides, lying inside allowed zones, children playing, occlusion and identity swaps.
- CI runs ruff, mypy and pytest on synthetic or redistributable fixtures only: never real cameras, Telegram, private recordings or third-party datasets.
- Evaluation runs locally (`fim eval`) and reports event-level recall, false alerts per hour and alert latency per dataset. Never claim clinical accuracy.
- Pipeline changes are measured with `fim benchmark` on a stated hardware class: analysis rate per camera, confirmation latency, CPU and memory.

## 9. Security

- Secrets live only in environment variables or the git-ignored `.env`; `.env.example` holds placeholders.
- No ports are published to the internet. Telegram uses outbound long polling. PostgreSQL and MediaMTX stay on the internal Docker network unless a variable explicitly enables LAN access.
- Containers run as non-root users.
- Treat downloaded weights, datasets and archives as untrusted: verify the SHA-256 from the manifest and never unpickle untrusted files.
- The bot answers only the configured chats.

## 10. Models, datasets and licenses

- The repository stays redistributable under Apache-2.0. Never copy GPL, AGPL or unlicensed code into it, and never commit weights or datasets.
- Installations choose models and datasets for detection quality. Any published, openly downloadable model or dataset may be used privately by the installer. Its terms are recorded in `MODELS.md` or `docs/DATASETS.md`, it is never redistributed, and its authors are cited.
- Datasets that require registration or request forms are out of scope.
- Weights are downloaded at image build time from the original source and verified against the SHA-256 in the model manifest. Nothing is downloaded at runtime.
- Weights trained on household recordings or on non-commercial or research-only data are never published.
- Every new dependency, model or dataset updates `THIRD_PARTY_NOTICES.md`, `MODELS.md`, `docs/DATASETS.md` or `CREDITS.md`, as appropriate.

## 11. Configuration

- Every main setting is an environment variable with the `FIM_` prefix, validated at start-up and documented both in the README variable table and in `.env.example`. A new variable without documentation is unfinished work.
- Cameras are declared through variables; adding a camera needs configuration only, never code.
- Zone files (hazard zones and allowed-lying zones) are referenced by variables and stored in git-ignored paths.
- A hash and a snapshot of the active configuration are stored with every event.

## 12. Docker Compose and PostgreSQL

- Everything runs in Docker. Supported commands go through Compose, e.g. `docker compose run --rm app fim …`; native `uv run` is optional for development.
- Services: `mediamtx` (official image, pinned version), `postgres` (official image, pinned major version) and `app` (built locally). Images are never published.
- Volumes: `pgdata`, `recordings`, `events`, `artifacts`. Every service has `restart: unless-stopped` and a health check.
- Updates: `git pull && docker compose up -d --build`. Alembic migrations run at app start-up. Keep the previous app image tagged for rollback and follow the README's on-site update checklist.
- PostgreSQL stores events, evidence summaries, notification states, feedback, the recording-segment index, health history and configuration snapshots. It never stores video or images. Timestamps are `timestamptz` in UTC.
- Migrations are reviewed in PRs. Backups follow the documented `pg_dump`/`pg_restore` procedure; test a restore after schema changes.

## 13. GitHub workflow

- Never push to `main`. One branch per issue: `feat/<issue>-<slug>`, `fix/…`, `docs/…`, `chore/…`, `test/…`.
- Conventional Commits (`feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `chore:`) in English, imperative mood.
- One PR per issue, linked with `Closes #N`. The description states what changed, how it was tested (with evidence), limitations and follow-ups.
- Merge only with green CI and maintainer approval.
- Issues are written in English with explicit acceptance criteria and a milestone (`MVP install` or `Post-MVP`).
- Never include private details in issues, PRs, screenshots or logs.

## 14. Documentation and ADRs

- `docs/ARCHITECTURE.md` describes the current design. Update it in the same PR as any design change.
- Decisions live in `docs/adr/NNNN-title.md` (Context, Decision, Alternatives considered, Consequences). Changes to the stack, services, policies or safety rules need a new ADR; accepted ADRs are superseded, never rewritten.
- The README holds the quick start, every environment variable, known limitations and the safety disclaimer.
- Keep `MODELS.md`, `docs/DATASETS.md`, `THIRD_PARTY_NOTICES.md`, `CREDITS.md` and `NOTICE` current.

## 15. Working agreements for coding agents

- Plan non-trivial work before coding. State assumptions, and ask the maintainer (in pt-BR) when a requirement is ambiguous.
- Prefer maintained open-source components, with credits, over new subsystems; do not adopt a large platform for a small feature.
- Keep changes scoped to the issue. Report unrelated problems instead of fixing them silently.
- Never weaken a safety or privacy rule to make a test pass or to save time.

## 16. Definition of Done

- [ ] The issue's acceptance criteria are met and demonstrated.
- [ ] Tests were written first where applicable and pass locally and in CI (ruff, mypy, pytest).
- [ ] Safety rules hold: no suppression by attribution, honest wording, degraded states reported, alerts independent of the database.
- [ ] New settings appear in the README table and in `.env.example`.
- [ ] Architecture, ADRs, `MODELS.md`, `docs/DATASETS.md`, `THIRD_PARTY_NOTICES.md` and `CREDITS.md` are updated when affected.
- [ ] Performance impact was measured if the pipeline changed.
- [ ] Limitations and open questions are documented.
- [ ] The diff contains no secrets, private media or household details.
- [ ] The maintainer reviewed and approved the PR.
