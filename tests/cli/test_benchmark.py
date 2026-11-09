from pathlib import Path

import numpy as np

from fall_impact_monitor.cli.benchmark import run_detector_benchmark
from fall_impact_monitor.perception.detector import Detection

from ..conftest import write_synthetic_video


class BoxDetector:
    """Always returns the labelled box: perfect recall."""

    def __init__(self, box: tuple[float, float, float, float]) -> None:
        self.box = box

    def detect(self, image_bgr: np.ndarray) -> list[Detection]:
        assert image_bgr.shape == (360, 640, 3)
        return [Detection(self.box, 0.9)]


class NothingDetector:
    def detect(self, image_bgr: np.ndarray) -> list[Detection]:
        return []


def make_dataset(root: Path) -> Path:
    folder = root / "Subject.1" / "Fall forward"
    folder.mkdir(parents=True)
    video = write_synthetic_video(folder / "FallForwardS1.mp4", frames=20)
    for i in range(1, 21):
        cls = 1 if i > 10 else 0
        (folder / f"cff1{i:05d}.txt").write_text(f"{cls} 0.5 0.5 0.25 0.5\n")
    return video


def test_benchmark_reports_recall_per_label_and_timing(tmp_path: Path) -> None:
    make_dataset(tmp_path)
    box = (240.0, 90.0, 400.0, 270.0)  # the labelled box in pixels
    rows = run_detector_benchmark(
        tmp_path, {"perfect": BoxDetector(box), "blind": NothingDetector()}, frames_per_video=6
    )
    by_name = {row.model: row for row in rows}
    assert by_name["perfect"].fall_recall == 1.0
    assert by_name["perfect"].other_recall == 1.0
    assert by_name["perfect"].frames == 6
    assert by_name["blind"].fall_recall == 0.0
    assert by_name["blind"].mean_ms >= 0.0
