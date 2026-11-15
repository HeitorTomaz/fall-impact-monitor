"""Person detection with YOLOX ONNX exports (raw grid output, no NMS in the graph)."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from fall_impact_monitor._vendor.yolox import demo_postprocess, nms, preproc

PERSON_CLASS = 0
COCO_COUCH_CLASS = 57
COCO_BED_CLASS = 59


class OnnxSession(Protocol):
    def get_inputs(self) -> list[Any]: ...
    def get_outputs(self) -> list[Any]: ...
    def run(self, output_names: Any, input_feed: dict[str, Any]) -> list[Any]: ...


def create_session(
    path: Path, providers: list[str] | None = None, threads: int | None = None
) -> OnnxSession:
    """``threads``: per session (see ``perception.threads``); ``None``: ONNX Runtime's own
    default, one per core."""
    import onnxruntime as ort

    from fall_impact_monitor.perception.threads import session_options

    session: OnnxSession = ort.InferenceSession(
        str(path),
        sess_options=session_options(threads) if threads else None,
        providers=providers or ["CPUExecutionProvider"],
    )
    return session


@dataclass(frozen=True)
class Detection:
    box_xyxy: tuple[float, float, float, float]
    score: float


class PersonDetector:
    """YOLOX detector of one COCO class (people by default). Input: BGR uint8 image.
    Output: boxes in image pixels."""

    def __init__(
        self,
        session: OnnxSession,
        *,
        score_threshold: float = 0.3,
        nms_iou: float = 0.45,
        class_id: int = PERSON_CLASS,
    ) -> None:
        model_input = session.get_inputs()[0]
        self._session = session
        self._input_name: str = model_input.name
        self._size = (int(model_input.shape[2]), int(model_input.shape[3]))
        self._score_threshold = score_threshold
        self._nms_iou = nms_iou
        self._class_id = class_id

    @classmethod
    def from_path(
        cls, path: Path, providers: list[str] | None = None, **kwargs: Any
    ) -> "PersonDetector":
        return cls(create_session(path, providers), **kwargs)

    def detect(self, image_bgr: NDArray[np.uint8]) -> list[Detection]:
        blob, ratio = preproc(image_bgr, self._size)
        output = self._session.run(None, {self._input_name: blob[None]})[0]
        predictions = demo_postprocess(np.array(output, np.float32), self._size)[0]
        scores = predictions[:, 4] * predictions[:, 5 + self._class_id]
        keep = scores >= self._score_threshold
        if not keep.any():
            return []
        cxcywh = predictions[keep, :4]
        boxes = np.empty_like(cxcywh)
        boxes[:, :2] = cxcywh[:, :2] - cxcywh[:, 2:4] / 2
        boxes[:, 2:] = cxcywh[:, :2] + cxcywh[:, 2:4] / 2
        boxes /= ratio
        height, width = image_bgr.shape[:2]
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, width)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, height)
        kept_scores = scores[keep]
        return [
            Detection(
                box_xyxy=(
                    float(boxes[i, 0]),
                    float(boxes[i, 1]),
                    float(boxes[i, 2]),
                    float(boxes[i, 3]),
                ),
                score=float(kept_scores[i]),
            )
            for i in nms(boxes, kept_scores, self._nms_iou)
        ]


class End2EndPersonDetector:
    """Detector for exports with decoding and NMS inside the graph (``dets``, ``labels``),
    such as the MMDeploy YOLOX exports. Same letterbox as YOLOX."""

    def __init__(self, session: OnnxSession, *, score_threshold: float = 0.3) -> None:
        model_input = session.get_inputs()[0]
        self._session = session
        self._input_name: str = model_input.name
        self._size = (int(model_input.shape[2]), int(model_input.shape[3]))
        self._score_threshold = score_threshold

    @classmethod
    def from_path(
        cls, path: Path, providers: list[str] | None = None, **kwargs: float
    ) -> "End2EndPersonDetector":
        return cls(create_session(path, providers), **kwargs)

    def detect(self, image_bgr: NDArray[np.uint8]) -> list[Detection]:
        blob, ratio = preproc(image_bgr, self._size)
        dets, labels = self._session.run(None, {self._input_name: blob[None]})[:2]
        height, width = image_bgr.shape[:2]
        detections = []
        for (x1, y1, x2, y2, score), label in zip(dets[0], labels[0], strict=True):
            if int(label) != PERSON_CLASS or score < self._score_threshold:
                continue
            detections.append(
                Detection(
                    box_xyxy=(
                        float(np.clip(x1 / ratio, 0, width)),
                        float(np.clip(y1 / ratio, 0, height)),
                        float(np.clip(x2 / ratio, 0, width)),
                        float(np.clip(y2 / ratio, 0, height)),
                    ),
                    score=float(score),
                )
            )
        return detections


def load_detector(
    path: Path,
    providers: list[str] | None = None,
    score_threshold: float = 0.3,
    threads: int | None = None,
) -> "PersonDetector | End2EndPersonDetector":
    """Pick the right wrapper from the model outputs (raw YOLOX grid vs ``dets``/``labels``)."""
    session = create_session(path, providers, threads)
    if "dets" in {output.name for output in session.get_outputs()}:
        return End2EndPersonDetector(session, score_threshold=score_threshold)
    return PersonDetector(session, score_threshold=score_threshold)
