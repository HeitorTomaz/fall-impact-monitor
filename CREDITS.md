# Credits

Fall Impact Monitor builds on ideas from the projects, papers and datasets listed here. No code from them is included in this repository; each one informed the design. Thanks to their authors.

Copied or adapted code is listed in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md), model weights in [`MODELS.md`](MODELS.md) and datasets in [`docs/DATASETS.md`](docs/DATASETS.md).

## Projects and papers

| Source | URL | What informed this project |
|---|---|---|
| OmniFall: From Staged Through Synthetic to Wild, A Unified Multi-Domain Dataset for Robust Fall Detection | <https://arxiv.org/abs/2505.19889>, <https://huggingface.co/datasets/simplexsigil2/omnifall> | The distinction between *fall* (the falling motion) and *fallen* (the state of lying on the ground after a fall), used to separate fall events from person-on-floor events |
| cs2mosa/fall_detection | <https://github.com/cs2mosa/fall_detection> | A dedicated "fall recovered" outcome when the person stands up again within a few seconds, which shaped how falls with immediate recovery are still reported |
| Senior-audio-events | <https://github.com/SuperGithubDuo/Senior-audio-events> | Mapping audio-tagger classes to impact-like sounds, and tuning thresholds against a target rate of false alerts per hour |
| Face-Touching-Behavior | <https://github.com/IIT-PAVIS/Face-Touching-Behavior> | Hand-to-face detection from pose, which informed the hand-to-head reaction cue |
| Human-Falling-Detect-Tracks | <https://github.com/GajuuzZ/Human-Falling-Detect-Tracks> | Reference design for skeleton-based fall detection on tracked people |
| mir_eval | <https://github.com/mir-evaluation/mir_eval> | Event matching with a tolerance window (`match_events`), the basis for event-level precision and recall |
| sed_eval | <https://github.com/TUT-ARG/sed_eval> | Definitions of event-based metrics for sound event detection, including onset tolerance |
| MediaMTX | <https://github.com/bluenviron/mediamtx> | Single RTSP relay per camera, segmented recording and time-range playback, which shaped the capture and recording design. It runs as a separate service from its official image |
| edgefallkit | <https://github.com/suharvest/edgefallkit> | The overall design of an edge fall-detection toolkit, including a configurable fall state machine with scenario tests and separate documentation of model assets and third-party notices. The planned adaptation of its state machine is tracked in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) |
