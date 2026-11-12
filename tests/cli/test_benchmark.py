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


def test_pipeline_summary_estimates_rates_and_cameras() -> None:
    import pytest

    from fall_impact_monitor.cli.benchmark import summarize_pipeline

    report = summarize_pipeline(
        decode_ms=[2.0] * 10,
        analysis_ms=[48.0] * 10,
        people=[1] * 10,
        source_fps=25.0,
        person_fps=5.0,
        cpu_s=4.0,
        wall_s=2.0,
        max_rss_mb=512.0,
        frame_size=(1280, 720),
    )
    assert report.max_analysis_fps == pytest.approx(20.0)  # 1000 / (2 + 48)
    assert report.full_rate_keeps_up is False  # 20 < 25 frames per second
    # One camera costs 25 x 2 ms decoding + 5 x 48 ms analysis = 290 ms per second.
    assert report.cameras_estimate == pytest.approx(1000 / 290)
    assert report.cpu_percent == pytest.approx(200.0)
    assert report.analysis_ms_p95 == pytest.approx(48.0)


class CountingPipeline:
    def __init__(self) -> None:
        self.sizes: list[tuple[int, int]] = []
        self.attention_until = -1.0

    def request_attention(self, until_s: float) -> None:
        self.attention_until = until_s

    def process(self, frame_bgr: np.ndarray, record: object) -> object:
        from fall_impact_monitor.api.perception import FrameResult

        assert record.arrival_ns / 1e9 <= self.attention_until  # type: ignore[attr-defined]
        self.sizes.append((frame_bgr.shape[1], frame_bgr.shape[0]))
        return FrameResult(record, [], analysed=True)  # type: ignore[arg-type]


def test_pipeline_benchmark_analyses_every_frame_at_the_service_size(tmp_path: Path) -> None:
    import pytest

    from fall_impact_monitor.cli.benchmark import run_pipeline_benchmark

    video = write_synthetic_video(tmp_path / "people.mp4", frames=30)
    pipeline = CountingPipeline()
    report = run_pipeline_benchmark(
        str(video), pipeline, max_frames=20, max_width_px=320, person_fps=5.0
    )
    assert report.frames == 20 and len(pipeline.sizes) == 20
    assert set(pipeline.sizes) == {(320, 180)}
    assert report.frame_size == (320, 180)
    assert report.source_fps == pytest.approx(25.0, rel=0.05)
