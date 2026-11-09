"""Build a perception pipeline from settings and the model manifest."""

from pathlib import Path

from fall_impact_monitor.api.perception import PerceptionPipeline
from fall_impact_monitor.config import Settings
from fall_impact_monitor.perception.cadence import CadencePolicy, RateMonitor
from fall_impact_monitor.perception.detector import load_detector
from fall_impact_monitor.perception.models import load_manifest, resolve
from fall_impact_monitor.perception.pose import PoseEstimator
from fall_impact_monitor.perception.tracking import PersonTracker


def build_pipeline(
    settings: Settings,
    *,
    started_s: float,
    manifest_path: Path = Path("models.toml"),
    providers: list[str] | None = None,
) -> PerceptionPipeline:
    """Verified models + tracker + cadence; raises `ModelError` naming a missing model."""
    manifest = load_manifest(manifest_path)
    models_dir = Path(settings.models_dir)
    detector = load_detector(
        resolve(manifest[settings.detector_model], models_dir),
        providers,
        score_threshold=settings.detector_score_threshold,
    )
    pose = PoseEstimator.from_path(resolve(manifest[settings.pose_model], models_dir), providers)
    return PerceptionPipeline(
        detector,
        pose,
        PersonTracker(frame_rate=settings.person_fps),
        CadencePolicy(empty_fps=settings.empty_fps, person_fps=settings.person_fps),
        RateMonitor(min_fps=settings.min_analysis_fps, started_s=started_s),
    )
