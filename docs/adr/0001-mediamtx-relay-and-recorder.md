# ADR-0001: MediaMTX as RTSP relay and recorder

- Status: Accepted
- Date: 2026-11-08

## Context

The system needs, per camera:

- a full-rate, stream-copy recording of the main stream (no transcoding) from which pre/post-event clips can be cut;
- a separate stream for analysis that may be decoded at a lower or adaptive rate;
- robust reconnection when the camera or Wi-Fi drops.

Consumer RTSP cameras (including the reference TP-Link Tapo C113) accept only a few simultaneous RTSP sessions. Opening one connection per consumer (recorder, analyzer, audio reader, occasional live view) would exhaust them and make behavior depend on how many tools are attached.

## Decision

- Run **MediaMTX** (MIT) as the `mediamtx` service, using the official image with a pinned version.
- MediaMTX opens a single upstream connection per camera stream and relays it internally to the `app` service.
- It records the camera **main stream** as fMP4 segments of about one minute, by stream copy, including audio when present.
- Event clips are extracted through MediaMTX's playback API (`/get` with start and duration), which returns one MP4 starting at the GOP that contains the requested start. The requested and the actually obtained windows are both stored.
- Retention (age, disk quota, later motion/person-based tiers) is managed by the `app`, not by MediaMTX's uniform deletion.
- MediaMTX configuration is derived from the same environment variables as the application (single source of truth); whether through `MTX_*` variables or its configuration API is an implementation choice.
- `useAbsoluteTimestamp` stays disabled: some cameras send unreliable RTCP sender-report clocks.
- Phase 1 includes a 24–48 h soak test against the reference camera. If the recorder proves unstable, the fallback is an `ffmpeg -c copy -f segment` recorder supervised by the app (a well-known recipe, credited when adapted).

## Alternatives considered

- **go2rtc** (MIT): good single-connection relay, but it does not record; recording and clip extraction would have to be built.
- **Frigate** (MIT): provides recording, retention and a review UI, but it is a full NVR platform; it keeps decoding a detect stream even with detection off, uses wall-clock timestamps in its RTSP presets, and couples us to its release cycle. Too large for the part we need.
- **Plain `ffmpeg -f segment` subprocess**: no extra service, but retention, indexing and clip assembly become our code, the CLI silently rewrites backward timestamps, and each process uses a camera session unless a relay exists anyway. Kept as plan B.

## Consequences

- One extra container (~30 MB image), multi-architecture (amd64/arm64).
- Clips start at a keyframe (GOP boundary), so the real window may begin a few seconds before the requested one; this is recorded, never hidden.
- Known risk: an open upstream report describes the H.264 recorder stopping on backward timestamps. The soak test and recorder health monitoring cover it.
- Camera stream limits are respected regardless of how many internal consumers exist.

## References

- https://github.com/bluenviron/mediamtx
- https://mediamtx.org/docs/features/record
- https://mediamtx.org/docs/features/playback
- https://github.com/bluenviron/mediamtx/pull/5714
