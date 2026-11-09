# Perception benchmark

Results on public datasets only, reproducible with the commands below. Hardware class: desktop x86_64 CPU, 4 cores available to Docker, ONNX Runtime CPU execution provider (1.31). Date: 2026-11-09.

## Person detectors on fallen people (CAUCAFall)

`fim benchmark detectors --dataset-root <CAUCAFall folder> --models yolox_tiny,yolox_s,yolox_m_humanart --frames-per-video 8`

100 videos (10 subjects × 5 falls and 5 activities), 8 labelled frames per video, half of them with a person labelled as fallen. Recall = labelled people matched by a detection (one-to-one), with the detector's score threshold at 0.3 (the same score now needed to start or continue a track).

| Model | Fall recall (IoU ≥ 0.5) | Other recall (IoU ≥ 0.5) | Fall recall (IoU ≥ 0.3) | ms per frame |
|---|---|---|---|---|
| YOLOX-Tiny (416) | 0.531 | 0.880 | — | 21 |
| YOLOX-S (640) | 0.625 | 0.913 | 0.651 | 78 |
| YOLOX-m HumanArt (640) | **0.766** | 0.906 | 0.766 | 115 |

The detector choice matters most for people on the floor. YOLOX-m HumanArt is the default; YOLOX-S is the alternative with permissively licensed training data.

## Observation: fallen people at the edge of the frame

On a public GMDCSA24 fall video, a person who ends up lying partly outside the frame was detected only weakly (scores 0.06–0.25) and never became a confirmed track: the box changes shape abruptly when a person falls, so box-overlap association fails, and weak detections cannot start new tracks. Therefore:

- the detection threshold defaults to 0.1 and every detection of an analysed frame is exposed in `FrameResult.detections` with its pose and track id (if any), so floor rules do not depend on tracking alone;
- tracks start and continue from detections scoring at least 0.3 (`FIM_TRACKER_NEW_TRACK_MIN_SCORE`; the tracker library default of 0.6 left 29 % of fallen people untracked), survive 3 s without detections, and lying people are estimated on 90-degree rotated crops;
- the fall rules (issue #4) must treat a track that disappears right after a fast descent as suspicious, never as a normal exit.

## Reproduce

Download the datasets yourself (see [`../DATASETS.md`](../DATASETS.md)), fetch the models (`fim models fetch`), then run the commands above through `docker compose --profile dev run --rm dev uv run --frozen …`.
