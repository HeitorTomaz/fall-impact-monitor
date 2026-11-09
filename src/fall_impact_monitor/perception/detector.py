"""Person detection with YOLOX ONNX exports (raw grid output, no NMS in the graph)."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from fall_impact_monitor._vendor.yolox import demo_postprocess, nms, preproc

PERSON_CLASS = 0


class OnnxSession(Protocol):
    def get_inputs(self) -> list[Any]: ...
    def run(self, output_names: Any, input_feed: dict[str, Any]) -> list[Any]: ...


def create_session(path: Path, providers: list[str] | None = None) -> OnnxSession:
    import onnxruntime as ort

    session: OnnxSession = ort.InferenceSession(
        str(path), providers=providers or ["CPUExecutionProvider"]
    )
    return session


@dataclass(frozen=True)
class Detection:
    box_xyxy: tuple[float, float, float, float]
    score: float


class PersonDetector:
    """YOLOX person detector. Input: BGR uint8 image. Output: boxes in image pixels."""

    def __init__(
        self, session: OnnxSession, *, score_threshold: float = 0.3, nms_iou: float = 0.45
    ) -> None:
        model_input = session.get_inputs()[0]
        self._session = session
        self._input_name: str = model_input.name
        self._size = (int(model_input.shape[2]), int(model_input.shape[3]))
        self._score_threshold = score_threshold
        self._nms_iou = nms_iou

    @classmethod
    def from_path(
        cls, path: Path, providers: list[str] | None = None, **kwargs: float
    ) -> "PersonDetector":
        return cls(create_session(path, providers), **kwargs)

    def detect(self, image_bgr: NDArray[np.uint8]) -> list[Detection]:
        blob, ratio = preproc(image_bgr, self._size)
        output = self._session.run(None, {self._input_name: blob[None]})[0]
        predictions = demo_postprocess(np.array(output, np.float32), self._size)[0]
        scores = predictions[:, 4] * predictions[:, 5 + PERSON_CLASS]
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
