# Fall Impact Monitor — System Architecture

> **Status:** PROPOSED ARCHITECTURE — subject to technical review  
> **Version:** 0.1 — 2026-11-08  
> **Repository:** https://github.com/HeitorTomaz/fall-impact-monitor  
> **License for original project source:** Apache License 2.0  
> **Primary implementation language:** Python  
> **Document language:** English

## 1. Purpose and decision status

Fall Impact Monitor is an independent, privacy-first, open-source application for local analysis of IP camera footage. Its purpose is to identify **possible falls, head impacts, and related hazardous movements**, and notify authorized responders.

The distinctive use case is a **possible head impact even without a complete fall or prolonged immobility**, such as a person stumbling into a wall, recovering balance, and continuing to walk.

This architecture documents the **current technical proposal**, not a final decision. Design alternatives, assumptions, complexity, and failure modes remain open for technical review. GitHub Issues #1–#9 define the current roadmap, subject to revision.

This is research and assistive software, **not a medically validated fall detector, head-injury detector, or emergency service**. It cannot rule out a concussion or intracranial injury. Initial deployment is supplementary to continuous human supervision; it must not imply that a person can safely be left unattended.

### 1.1 Goals

1. Analyze videos **locally** without streaming raw camera footage to external cloud services.
2. Detect falls **including immediate recovery**, without requiring a person to remain on the floor.
3. Investigate possible **head impacts without a fall**, including lateral contacts and collisions with furniture or walls.
4. Support multiple visible persons and suppress child-attributed events **only when attribution is sufficiently reliable**. Uncertain cases must continue to alert.
5. Record reproducible evidence and use licensed public datasets for transparent evaluation.
6. Notify authorized contacts through a pluggable notification adapter (initially Telegram), with acknowledgment and manual feedback.
7. Run on existing commodity CPU-first hardware; permit optional acceleration and eventual x86_64/ARM64 portability.
8. Keep all private installation details, camera credentials, and household footage out of the public repository.
9. Allow new cameras and sensors later without a fundamental rewrite.

### 1.2 Non-goals for the initial MVP

- Medical diagnosis, clinical-grade accuracy, emergency-service integration, or unattended safety guarantees.
- Face recognition or mandatory identity matching to one named person.
- A custom Android/iOS application, cloud video inference, training a foundation model, or a complex microservice deployment.
- Capturing private spaces where consent and appropriate privacy controls cannot be maintained. Privacy-preserving sensors may be evaluated separately.
- Mobile or outdoor fall monitoring; wearable sensors are a possible complementary integration.
- Mandatory use of a GPU or mandatory buying of new processing hardware.
- Distribution of private footage or third-party datasets that cannot lawfully be redistributed.

### 1.3 Reference platforms and deployment independence

The software targets **generic RTSP-compatible IP cameras**, with optional adapters for additional sources. Platform characteristics must be configurable; deployment-specific settings and test observations belong in local, untracked deployment notes rather than this public architecture.

| Dimension | Public reference or architectural requirement |
| --- | --- |
| Reference test camera | **TP-Link Tapo C113** (Wi-Fi IP camera; RTSP / H.264), used for integration and performance testing |
| Input compatibility | RTSP network streams and local MP4, MKV and AVI recordings |
| Video configuration | Discover resolution, codec, frame timing and frame rate at runtime instead of assuming fixed settings |
| Camera count | One or more independent camera inputs with configurable per-camera resources |
| Inference host | CPU-compatible x86_64 or ARM64 where supported; GPU or hardware decoding is optional |
| Operating systems | Cross-platform development and testing; Linux local deployment recommended but not required |
| Network | Local IP connectivity, preferably isolated and without external camera exposure |
| Notifications | Configurable outbound notification adapter; initial implementation may use Telegram |
| Storage | Bounded local retention, no mandatory cloud upload |

**Compatibility note:** Naming the Tapo C113 documents a concrete device developers can reproduce tests with; it does not make that camera a prerequisite for using the project.

## 2. Fundamental constraints and safety rules

- **A visible collision is not a measured head impact.** A 2D monocular camera cannot reliably distinguish passing in front of a wall from striking it, particularly with occlusion, blur, reflections, and limited viewpoint.
- A normal-looking person after a possible hit may still have a serious injury; the software must never label someone "medically cleared".
- If the system fails or is uncertain about person association, **do not silently suppress a potentially adult-related event**.
- Distinguish observed evidence, confidence, inference, and actual diagnosis in event labels.
- Unavailable video, frame timing anomalies, stalled inference, and lost connectivity are **degraded monitoring**, not "no incident".
- Raw video does not leave the premises by default; event metadata and messages are the only expected outbound data.
- All activity must be consent-based, privacy-conscious, and understood by caregivers and other occupants.

## 3. Proposed logical topology

~~~mermaid
flowchart TD
    CAM["IP Wi-Fi cameras / RTSP"] --> ING["Per-camera ingest / decode / timebase"]
    ING --> BUF["Encoded circular video buffer"]
    ING --> SCHED["Adaptive inference scheduler"]
    SCHED --> DET["Person detection and multi-person tracking"]
    DET --> POSE["Pose and head trajectory observations"]
    POSE --> FALL["Temporal fall detector"]
    POSE --> HEAD["Experimental head-impact candidate detector"]
    DET --> CHILD["Conservative child attribution"]
    FALL --> FUSE["Event fusion and deduplication"]
    HEAD --> FUSE
    CHILD --> FUSE
    FUSE --> EV["PostgreSQL event metadata store"]
    FUSE --> BUF
    FUSE --> ALERT["Telegram notification gateway"]
    ALERT --> ACK["Acknowledgment / responder state / feedback"]
    ACK --> EV
    HEALTH["Health metrics and watchdog"] --> EV
    HEALTH --> ALERT
    EV --> REPORT["Local evaluation and audit tools"]
    EV -.-> REMOTE["Optional remote monitoring endpoint"]
~~~

**Important separation:** the recording path must preserve the camera's high-frequency encoded video even if the inference path works at fewer analyzed frames per second. Inference results and video availability are not the same metric.

## 4. Proposed repository boundaries

The implementation should be a small modular Python application. These are suggested packages, not requirements to create all folders before Phase 1.

~~~text
fall-impact-monitor/
  README.md
  LICENSE
  .gitignore
  .env.example
  pyproject.toml
  compose.yaml        # Local deployment stack
  Dockerfile          # Application container
  config/
    example.yaml
  src/fall_impact_monitor/
    capture/         # sources, ingest, decoding, timebase
    buffering/       # bounded encoded circular capture and clip export
    tracking/        # person detections, track IDs, pose estimation
    detectors/       # independent fall and possible impact analyzers
    events/          # event fusion, state, persistence, feedback
    notifications/   # Telegram integration and acknowledgment
    health/          # runtime health and fault recovery
    evaluation/      # dataset manifests, validation, metrics
    cli/             # operator and developer commands
  tests/
    unit/
    integration/
    fixtures/       # only synthetic or redistributable examples
  docs/
    ARCHITECTURE.md
    DATASETS.md
    SECURITY.md
    DECISIONS.md
  .github/
    workflows/
  test_videos/       # LOCAL ONLY, ignored by Git
  local/             # LOCAL ONLY, optional deployment notes; ignored by Git
  artifacts/         # LOCAL ONLY, ignored by Git
~~~

A single host/process or a few local processes are acceptable at the start. **Docker Compose** is the default local deployment orchestrator, with an application service and a PostgreSQL service, both on a private Docker network. Do not add message brokers, distributed tracing, dashboards, Kubernetes, or cloud dependencies without a measured need.

## 5. Capture, video timing and buffering

### 5.1 Inputs

The common input abstraction must support:

- RTSP high-quality stream (stream1) from compatible IP cameras.
- Local MP4, MKV and AVI files for reproducible experiments.
- Synthetic frame and timestamp sequences for automated tests.
- Multiple cameras, eventually with one ingest session per camera.
- Optional lower-resolution substream for presence checks, if useful and available.

Prefer mature capture components such as FFmpeg/PyAV, considering GStreamer or an RTSP relay such as go2rtc **only if measured tests justify additional complexity**. Do not build a custom video decoder.

### 5.2 Time domains and dropped-frame accounting

A frame record should, when available, carry:

- camera/source identifier and frame sequence identifier;
- encoded stream presentation timestamp (PTS), decode timestamp (DTS), and rational time base;
- monotonic clock time when the frame reaches the application;
- wall-clock UTC time for logs and human-facing event timestamps;
- decode completion time, inference submission time, inference completion time;
- actual decoded dimensions and frame format;
- flags indicating missing, repeated, non-monotonic or unreliable source timestamps.

These time sources are **not interchangeable**. In particular, monotonic local reception time captures arrival jitter, not necessarily the precise physical instant of a scene event. Use source presentation times for kinematics only after validation; maintain explicit uncertainty when timestamps cannot be trusted. Never calculate physical speed with a zero or negative time delta. Never silently invent "missing original frames" based only on nominal FPS.

Track separately:

1. Stream-reported nominal FPS.
2. Decoded frames/sec.
3. Frames delivered to the application/sec.
4. Frames selected for inference/sec.
5. Inferences completed/sec and their latency.
6. Gap and jitter distributions based on appropriate time domains.
7. Explicit drops or late frames observed *inside the application*, vs inferred source losses.

### 5.3 Capture validation and camera compatibility

Test the **TP-Link Tapo C113** as the initial reference hardware, while retaining a device-independent RTSP ingest layer. Public documentation should describe the camera model, compatible protocols and reproducible, anonymized test methods, without disclosing private installation details or private capture files.

Capture validation should compare:

- stream metadata, source PTS/DTS and actual decode order;
- frame count, inter-frame timing and recorded container time base;
- remux, decode/re-encode and playback behavior, including FFmpeg and VLC where useful;
- buffering, concurrent clients, network jitter and reconnect behavior;
- motion continuity using synthetic, licensed or deliberately non-identifying test scenes.

**Do not infer physical sensor frame rate solely from container metadata.** The presence of a camera model in a reference benchmark does not establish accuracy across firmware revisions, lighting conditions, or networks. Publish only reproducible results that contain no household identifiers, addresses, credentials or private footage.

### 5.4 Storage strategy

Do **not** keep a circular buffer of full-resolution decoded RGB frames. That would consume excessive RAM (several hundred megabytes per second per camera at these dimensions). Prefer bounded **encoded video segments** with keyframe-aware extraction.

Initial **proposed**, configurable defaults:

| Parameter | Initial value |
| --- | --- |
| Ring buffer per camera | At least 60 seconds |
| Pre-event clip | 20 seconds |
| Post-event clip | 60 seconds |
| Event clip retention | 7 days |
| Default continuous archive | Disabled |
| Outbound video upload | Disabled |
| Local disk quota | Configurable, with cleanup before disk exhaustion |

For live events, segment boundaries and H.264 keyframes may affect the exact clip start. Store the requested window alongside the actual extracted window; never claim perfect cut accuracy without validating it.

## 6. Inference and individual tracking

### 6.1 Person and pose pipeline

Proposed interfaces:

- Person detector outputs boxes and confidence by frame.
- Multi-person tracker outputs ephemeral per-camera track IDs with visibility/occlusion flags.
- Pose estimator outputs joints, confidence, head-related landmarks and camera-relative coordinates.
- Temporal analyzer consumes a **time-ordered per-person track**, not merely a global frame classification.

MediaPipe Pose Landmarker is a candidate baseline, not an imposed final choice. Review model quality, license of code **and weights**, CPU speed, multiple-person support, occlusion behavior and exportability. The Python live-stream mode may drop submissions when busy; the application must instrument requested versus completed inference. Evaluate other permissively usable pose and tracking solutions during technical review.

### 6.2 Adaptive processing

Maintain full-rate ingest and encoded recording; avoid assuming the CPU can infer all frames at camera FPS.

- Empty/static scene: reduce heavy inference adaptively **after** verifying no person or suspicious event requires tracking.
- Person visible: sustain an appropriate tracking/inference cadence and detect sudden movement.
- Possible fall, head movement anomaly, or suspicious occlusion: temporarily increase attention to available frames and preserve surrounding video.
- After a potential fall, keep observing even if the scene becomes static: **post-fall immobility is an important signal**.
- Preserve provenance of which frames were analyzed; never claim skipped frames were analyzed.
- Rate control must be benchmark-driven for one and three camera inputs.

### 6.3 Multiple people and children

Child-related motion is a false-alert source for the target use case. However, age/size inference is uncertain and perspective-dependent.

Policy:

1. Track individuals separately and associate candidate events with a specific track whenever possible.
2. Suppress a child-attributed event **only with high confidence and validated performance**.
3. If identity is unknown, tracking switches, a small adult is mistaken for a child, or an adult is occluded, **do not suppress the alert**.
4. An adult suffering an impact due to a child nearby must still generate an adult-associated event.
5. Do not require face recognition; avoid biometric identity persistence.
6. Enable child suppression only after separate tests establish that adult events are not missed.

Consider a lower-priority event or review-only channel for uncertain events instead of indiscriminate hard filtering.

## 7. Event detection models

### 7.1 Fall detector

Detect abrupt change in support/posture, vertical displacement, body orientation, motion trajectory and context. **Do not require sustained immobility.** A person may fall and stand up immediately. Distinguish falls from sitting quickly, lying down, bending, playing, or slipping without falling.

A temporal state machine may use: NORMAL → SUSPECTED → EVENT → RECOVERY / CLOSED. An event cannot be erased merely because a person has recovered.

### 7.2 Possible head-impact detector (experimental)

Independent of the fall detector. Combine evidence from head landmark trajectories, relative motion to trunk, rapid direction changes/deceleration, proximity to known surfaces, occlusion, and optional cross-camera observations later.

Do not claim that monocular pixel coordinates measure physical head-to-wall distance or impact force. Potential output categories:

- possible_head_impact: concerning evidence requiring human checking;
- fall_detected: a fall sequence independently identified;
- near_miss_or_imbalance: lower-priority hazardous movement;
- uncertain_motion: ambiguous case retained for review;
- person_down_or_immobile: post-event observation, not a prerequisite.

Choose thresholds on held-out data; record uncertainty, false alarms and misses. A safe staged near-wall movement is not ground truth for a real head impact.

### 7.3 Fusion, timing, priority and deduplication

The fusion engine merges overlapping per-person evidence without allowing a low-confidence class (e.g., child) to override high-risk adult evidence. Events receive immutable IDs, associated camera IDs, track IDs if reliable, time window, type, score/evidence, review state, and optional clip reference.

Initial alerting rules are policy-configurable; a possible head impact should be alerted promptly even if no fall is detected. Persist the original evidence even if alert classification is subsequently corrected.

## 8. Event storage and notifications

### 8.1 Local state

**PostgreSQL is the primary database**, deployed as a dedicated service through Docker Compose. This is a deliberate portability and operational-maintenance choice, not a claim that embedded SQLite databases are inherently unreliable. PostgreSQL stores structured event state, notification attempts, and clip metadata; it does **not** store video blobs. Suggested conceptual entities:

- CameraStatus (camera ID, last decoded frame, health status).
- DetectionObservation (timestamps, inference version, track ID, evidence).
- IncidentEvent (ID, time range, category, evidence, status, camera).
- ClipMetadata (path, requested/actual windows, retention expiry, checksum).
- NotificationAttempt (event, destination, send outcome, timestamp).
- ResponderAcknowledgment (event, responder alias, acknowledged / attending / resolved).
- Feedback (false positive, confirmed incident, manually reported missed event).
- ServiceHealth (last heartbeat, metric snapshots, faults).

Do not persist full images or biometrics by default. Apply schema migrations (SQLAlchemy and Alembic are proposed Python tooling) and storage quotas; test behavior when disk becomes read-only/full. PostgreSQL should have a dedicated persistent volume, a private network without public database-port exposure, appropriate health checks, and tested backup **and restore** procedures. Private event clips should use a separate bounded local filesystem or volume. Restart and reconnection logic must tolerate a database becoming unavailable after container startup; service startup ordering alone is insufficient.

### 8.2 Telegram workflow

Telegram Bot API is the initial remote notifier. The bot should use a narrowly authorized group and no raw video upload by default.

Lifecycle: DETECTED → NOTIFIED → ACKNOWLEDGED / ATTENDING → REVIEWED / RESOLVED. If nobody acknowledges, a configurable reminder/escalation policy may retry or use a future independent channel. Telegram delivery does not prove that anyone saw or acted on the alert.

Support manual incident reports and false-positive review. Protect the token, Telegram chat identifiers, and callback endpoints. Prefer outbound-only connectivity; do not expose camera ports publicly.

### 8.3 Failure visibility

Monitor camera disconnection, stale stream, decoding failures, queue backlog, excessive inference latency, process restarts, disk consumption, and inability to send notifications. Optional **remote heartbeat**, using a minimal endpoint, is useful to detect loss of the entire host; no cloud account is required for core detection.

If the internet is down, continue local detection and queue events when practical; remote Telegram notification will not work during that outage. Future local audible alarms, UPS, cellular fallback, and alternate notification channels are possible improvements.

## 9. Security, privacy and open-source hygiene

- All credentials must come from environment variables, local secret files, or an OS secret manager; never hard-code them.
- Never commit test_videos/, artifacts/, private footage, debug snapshots, .env secrets, telemetry dumps, IP-specific config or family-specific identifiers.
- Security defaults should avoid exposure of RTSP/ONVIF ports to the public internet. Remote support should use an authenticated VPN when needed.
- Apply least-privilege file permissions, pin dependencies, use package vulnerability checks, and review external PR code before CI executes it.
- CI must use synthetic/redistributable fixtures only and no real camera, Telegram credentials or private environment.
- Treat contributor-submitted model weights, pickle files, archives and datasets as untrusted inputs.
- Avoid publishing identifiable people, especially children, without appropriate permissions.
- Public docs, examples and issues must remain generic and independent of any unrelated product or organization.
- Provide SECURITY.md, responsible disclosure instructions, and transparent limitations.

## 10. Dataset, model and dependency licensing

**The repository's Apache-2.0 license covers original project code; it does not automatically relicense third-party video datasets or model weights.** Each asset has separate rights and obligations.

A dataset may be approved only after checking its authoritative license and documenting permissions for:

1. Download and access.
2. Processing/annotation and modification.
3. Training and validation on the dataset.
4. Commercial reuse of the trained model and inference outputs.
5. Redistribution of datasets or derived assets, if contemplated.
6. Publishing resulting model weights, if contemplated.
7. Attribution, privacy/consent, derivative licensing, and notices.

Prefer clearly permissive datasets such as CC0 or CC BY 4.0 (with required attribution) or equivalent compatible terms. **Do not approve** non-commercial, research-only, no-derivatives, no-training, no-commercial-inference, or ambiguously licensed sources for the default pipeline. Custom/share-alike terms need explicit review. Public availability is not a license.

Maintain a versioned manifest and docs/DATASETS.md with source, version, canonical license URL, usage permissions, known restrictions, approval status and credit. CI/default scripts must refuse unapproved dataset identifiers. **Dataset licenses, model-weight licenses and code licenses must be audited independently**. Issue #2 is the policy source of truth.

Datasets should be downloaded explicitly by the user and stored outside Git. Splits must prevent data leakage by person, scene and near-duplicate versions. Provide a small fully synthetic CI fixture.

## 11. Quality gates, evaluation and safety

Separate functional smoke tests, model evaluation and real-world validation.

| Category | Required evidence |
| --- | --- |
| Capture | RTSP/file input, timing report, gap metrics, reproducible FFmpeg vs VLC comparison |
| Pose/tracking | Multi-person ID consistency, occlusion failure cases, throughput |
| Falls | Per-event recall and false alarms on held-out licensed videos, including stand-up recovery |
| Head-impact candidates | Separate recall/false alarms for no-fall scenarios; explicit limitations and uncertain labels |
| Children | Child-event suppression accuracy and critically, adult-event recall while children are present |
| Alerts | End-to-end detection-to-notification latency, ACK state, delivery failures |
| Reliability | Camera restart/unplug tests, internet outage, disk limits, process recovery, prolonged run |
| Portability | CPU-only path and Linux test; acceleration optional, not mandatory |
| Public release | No secrets/private footage; provenance and licenses documented; PR review |

Use scenario-level and event-level measurements, not just frame accuracy. Track false alerts per camera-hour, missed events, time-to-detection, time-to-notification and downtime. Define thresholds based on evidence after collecting representative data; do not invent medical-grade performance targets.

No dangerous demonstrations with older adults or children. Use publicly approved recordings and safely staged adult simulations or suitable props. Human caregivers remain responsible for assistance and medical decisions.

## 12. Execution environments and performance

### 12.1 Development

- Support development on mainstream desktop operating systems (Windows and Linux at minimum).
- WSL2 is an optional development convenience. Docker Compose is the standard local integration environment; capture and unit tests should also support execution without a host GPU or live database wherever practical.
- Tests must run on synthetic fixtures and licensed local files without live camera access.
- Local test media and deployment notes must remain outside version control.
- CLI and configuration must accept relative paths and environment-variable overrides.

### 12.2 Local deployment profiles

- **CPU-only baseline:** no dedicated GPU requirement; benchmark a representative low-power and desktop CPU.
- **Optional acceleration:** hardware decode or inference may be enabled through clearly isolated adapters after compatible hardware and licensing are verified.
- **Linux deployment:** Docker Compose with at minimum `app` and `postgres` services, health checks, restart policy, persistent database/video volumes, disk quotas and documented backup/restore. Optional services (RTSP relay, backup scheduler) require a concrete need.
- **Multi-camera mode:** scale by measured throughput and queue latency; avoid assuming linear performance.
- **Portable profiles:** retain an architecture that can adapt to x86_64 and ARM64 hardware without replacing core event logic.

Publish anonymized benchmarks identifying hardware **class**, software version and camera model. Keep any contributor-specific workstation configuration, network topology and operational secrets in ignored local documentation. Measure sustained workload, temperature, power, frame drops, latency and recovery before declaring a deployment operational.

## 13. Delivery plan and dependencies

| Phase / issue | Scope | Expected outcome |
| --- | --- | --- |
| #1 | Capture and timestamps | Reliable file/RTSP input, timing diagnosis and reproducible metrics |
| #2 | Approved datasets and evaluation | License registry, split strategy, CI fixture and repeatable reports |
| #3 | People, tracking and pose | Per-person pose tracks, CPU-first benchmark |
| #4 | Falls including quick recovery | Measured fall detector |
| #5 | Events, ring buffer and Telegram | End-to-end notification and feedback |
| #6 | Possible head impacts without a fall | Experimental independent head-event detector |
| #7 | Children and multi-person suppression | Conservative suppression with adult recall tests |
| #8 | Reliability and pilot | Fault recovery, limitations, supervised testing |
| #9 | Portable local deployment | Validated local operation on resource-constrained hardware where feasible |

The project roadmap is milestone-driven rather than tied to any one contributor's schedule. If a no-fall impact detector has not been validated, explicitly mark it experimental or disabled instead of suggesting it is safe.

## 14. Open architecture decisions

1. Can a single stream/decoder feed both a low-latency inference branch and loss-aware encoded event buffer without synchronization mistakes?
2. Is PyAV/FFmpeg the simplest reliable initial capture implementation? When would GStreamer/go2rtc/Frigate be preferable?
3. What source-time or local-time combination is trustworthy enough for short head-acceleration windows?
4. Which model and weights have a verifiable permissive license, workable CPU throughput and robust multi-person tracking?
5. Should initial child handling be only track-level event association, with automated age/size suppression postponed?
6. What evaluation corpus can be lawfully used, especially for **no-fall head impacts**?
7. How should uncertainty change notification priority without overwhelming caregivers?
8. What is the smallest storage/event architecture with safe behavior across restart, failed uploads and disk pressure?
9. What measurable release criteria can be achieved when some experimental components remain uncertain?
10. What should be removed from the MVP to minimize unnecessary work while preserving its key goal?

## 15. References (starting points, not approvals)

- TP-Link official RTSP/ONVIF integration: https://www.tp-link.com/us/support/faq/2680/
- TP-Link official RTSP FAQ: https://www.tp-link.com/us/support/faq/4465/
- MediaPipe Pose Landmarker Python guide: https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker/python
- MediaPipe Python API timestamp behavior: https://developers.google.com/edge/api/mediapipe/python/mp/tasks/vision/PoseLandmarker
- GitHub issues: https://github.com/HeitorTomaz/fall-impact-monitor/issues

References to third-party code or data are for investigation. **Do not incorporate a dependency, model or dataset without an independent license/provenance review.**
