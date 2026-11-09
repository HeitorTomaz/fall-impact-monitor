from pathlib import Path
from typing import Any

import numpy as np
import pytest

from fall_impact_monitor._vendor.yolox import demo_postprocess, nms, preproc
from fall_impact_monitor.perception.detector import Detection, PersonDetector

MODELS = Path(__file__).parents[2] / "models"


def test_preproc_places_image_top_left_and_pads_with_114() -> None:
    image = np.full((100, 200, 3), 7, np.uint8)
    blob, ratio = preproc(image, (416, 416))
    assert blob.shape == (3, 416, 416) and blob.dtype == np.float32
    assert ratio == pytest.approx(416 / 200)
    assert blob[0, 0, 0] == 7  # top-left is image
    assert blob[0, 415, 415] == 114  # bottom-right is padding


def test_demo_postprocess_decodes_grid_and_strides() -> None:
    out = np.zeros((1, 3549, 85), np.float32)
    decoded = demo_postprocess(out.copy(), (416, 416))
    assert decoded[0, 1, :4].tolist() == [8.0, 0.0, 8.0, 8.0]  # 2nd cell of stride-8 grid
    assert decoded[0, 52 * 52, :4].tolist() == [0.0, 0.0, 16.0, 16.0]  # first stride-16 cell


def test_nms_keeps_the_highest_scoring_overlapping_box() -> None:
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], np.float32)
    keep = nms(boxes, np.array([0.5, 0.9, 0.8], np.float32), 0.45)
    assert sorted(keep) == [1, 2]


class FakeInput:
    name = "images"
    shape = (1, 3, 416, 416)


class FakeSession:
    """Returns one person and one bicycle at stride-8 cell 1 (x=8, y=0)."""

    def get_inputs(self) -> list[FakeInput]:
        return [FakeInput()]

    def run(self, outputs: Any, feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
        assert feeds["images"].shape == (1, 3, 416, 416)
        out = np.zeros((1, 3549, 85), np.float32)
        out[0, 1, :5] = [0.5, 0.5, np.log(4.0), np.log(8.0), 0.9]  # cx=12, cy=4, w=32, h=64
        out[0, 1, 5] = 0.9  # person
        out[0, 10, :5] = [0, 0, 0, 0, 0.9]
        out[0, 10, 6] = 0.99  # bicycle, not a person
        return [out]


def test_detector_returns_people_only_in_image_coordinates() -> None:
    detector = PersonDetector(FakeSession(), score_threshold=0.3)  # type: ignore[arg-type]
    image = np.zeros((208, 208, 3), np.uint8)  # ratio 2.0
    detections = detector.detect(image)
    assert len(detections) == 1
    det = detections[0]
    assert det.score == pytest.approx(0.81)
    assert det.box_xyxy == pytest.approx((0.0, 0.0, 14.0, 18.0))  # clipped at the left/top


def test_empty_frame_returns_no_detections() -> None:
    class Empty(FakeSession):
        def run(self, outputs: Any, feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
            return [np.zeros((1, 3549, 85), np.float32)]

    assert PersonDetector(Empty()).detect(np.zeros((50, 50, 3), np.uint8)) == []  # type: ignore[arg-type]


@pytest.mark.skipif(not (MODELS / "yolox_s.onnx").exists(), reason="model weights not available")
def test_real_model_finds_a_person(synthetic_person_frame: np.ndarray) -> None:
    detector = PersonDetector.from_path(MODELS / "yolox_s.onnx")
    assert isinstance(detector.detect(synthetic_person_frame), list)
    assert all(isinstance(d, Detection) for d in detector.detect(synthetic_person_frame))
