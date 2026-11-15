"""Build a perception pipeline from settings and the model manifest."""

from pathlib import Path

from fall_impact_monitor.api.perception import PerceptionPipeline
from fall_impact_monitor.config import Settings
from fall_impact_monitor.perception.cadence import CadencePolicy, RateMonitor
from fall_impact_monitor.perception.detector import load_detector
from fall_impact_monitor.perception.models import load_manifest, resolve
from fall_impact_monitor.perception.pose import PoseEstimator
from fall_impact_monitor.perception.threads import available_cpus, inference_threads
from fall_impact_monitor.perception.tracking import PersonTracker


def build_pipeline(
    settings: Settings,
    *,
    started_s: float,
    cameras: int = 1,
    manifest_path: Path = Path("models.toml"),
    providers: list[str] | None = None,
) -> PerceptionPipeline:
    """Verified models + tracker + cadence; raises `ModelError` naming a missing model.
    ``cameras`` is how many pipelines share the CPUs (for the inference threads)."""
    manifest = load_manifest(manifest_path)
    models_dir = Path(settings.models_dir)
    threads = inference_threads(settings.inference_threads, cameras=cameras, cpus=available_cpus())
    detector = load_detector(
        resolve(manifest[settings.person_detector_model], models_dir),
        providers,
        score_threshold=settings.person_detector_min_score,
        threads=threads,
    )
    pose = PoseEstimator.from_path(
        resolve(manifest[settings.pose_estimator_model], models_dir), providers, threads=threads
    )
    return PerceptionPipeline(
        detector,
        pose,
        PersonTracker(
            frame_rate=settings.analysis_person_visible_fps,
            new_track_min_score=settings.tracker_new_track_min_score,
            track_lost_s=settings.tracker_lost_track_timeout_s,
        ),
        CadencePolicy(
            empty_fps=settings.analysis_empty_scene_fps,
            person_fps=settings.analysis_person_visible_fps,
        ),
        RateMonitor(min_fps=settings.analysis_min_fps, started_s=started_s),
        person_hold_s=settings.analysis_person_hold_s,
        detector_max_fps=settings.person_detector_max_fps,
    )
