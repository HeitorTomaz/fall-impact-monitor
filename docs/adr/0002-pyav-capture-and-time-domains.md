# ADR-0002: PyAV capture and explicit time domains

- Status: Accepted
- Date: 2026-11-08

## Context

Fall and head-motion analysis depends on correct frame order and time deltas over windows of a few hundred milliseconds. Preliminary local captures showed irregular source timestamps (runs of identical timestamps followed by forward jumps), and a container without per-frame timestamps can look smoother while discarding real timing. Consumer camera timestamp behavior also changes between firmware versions.

The capture layer must read local files and RTSP through one interface, expose packet-level timing, survive disconnections, and never fabricate data.

## Decision

- Use **PyAV** (BSD-3) for both file and RTSP input. It exposes per-packet `pts`, `dts`, `time_base` and side data, demuxer options (`rtsp_transport=tcp`, `allowed_media_types`) and open/read timeouts. No custom decoder.
- Keep three **time domains** separate and never interchange them:
  1. source presentation time (camera RTP-derived PTS);
  2. local monotonic arrival time (`monotonic_ns`);
  3. wall-clock UTC, for humans and logs only.
- Every decoded frame produces a `FrameRecord` (camera, sequence, pts, time base, arrival monotonic ns, wall UTC, keyframe flag, size) with a **timestamp-quality flag**: `ok`, `zero_delta`, `backward`, `outlier_gap` or `missing`.
- **Never invent timestamps or frames.** Frames with zero, negative or implausible deltas are flagged and excluded from kinematics; detection by box/pose continues with reduced confidence.
- The `app` analyzes the camera **substream** (via the MediaMTX relay) when available; without one it uses the main stream at reduced resolution. Audio is read in a separate session.
- An in-memory, time-bounded **encoded packet buffer** (~20 s, configurable) keeps recent packets with their timing metadata so the confirmation stage (ADR-0004) can re-decode a window at full rate.
- Reconnection uses exponential backoff; a stalled read or frozen PTS marks the stream stale and is reported as degraded monitoring.
- **Phase 1 decides which time domain kinematics use**, based on a reproducible protocol: direct vs relayed stream, main vs substream, with and without `allowed_media_types=video`, wall-clock stamping, and a ground-truth test (filming a stopwatch or blinking LED in bright and dim light at several frame rates).

## Alternatives considered

- **OpenCV `VideoCapture`**: no packet-level timestamps (only frame-unit PTS), opaque internal buffering, `read()` can hang on network loss, no reconnection. Usable only for image operations.
- **GStreamer** (`rtspsrc` jitter buffer, RTCP sync): powerful, but Python bindings are impractical on Windows development machines and add a large dependency; reconsider only if measurements justify it.
- **Wall-clock stamping of every frame** (`use_wallclock_as_timestamps`): hides source behavior and adds network jitter; kept only as a diagnostic mode.

## Consequences

- Kinematic evidence carries explicit timing confidence; poor timing degrades confidence rather than producing false velocities.
- PyAV wheels bundle FFmpeg builds that include GPL encoders; this matters only if binaries are redistributed, which the project does not do (ADR-0008, ADR-0009).
- The packet buffer costs only a few megabytes per camera.

## References

- https://pyav.basswood.io/docs/stable/api/packet.html
- https://ffmpeg.org/ffmpeg-protocols.html#rtsp
