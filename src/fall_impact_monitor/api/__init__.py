"""Public facade of the detection core. Must not depend on outputs or storage."""

from fall_impact_monitor.api.build import build_pipeline
from fall_impact_monitor.api.perception import DetectedPerson, FrameResult, PerceptionPipeline
from fall_impact_monitor.perception.tracking import Track

__all__ = ["DetectedPerson", "FrameResult", "PerceptionPipeline", "Track", "build_pipeline"]
