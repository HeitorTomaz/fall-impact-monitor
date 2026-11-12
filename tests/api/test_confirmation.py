from datetime import UTC, datetime
from fractions import Fraction
from typing import Any

import numpy as np
import pytest

from fall_impact_monitor.analysis.fall import FallAnalyzer
from fall_impact_monitor.api.confirmation import FallConfirmer, PendingConfirmations
from fall_impact_monitor.api.perception import FrameResult
from fall_impact_monitor.capture import FrameRecord, TimestampQuality
from fall_impact_monitor.events import EventCategory, make_event
from fall_impact_monitor.perception.tracking import Track
from tests.analysis.test_fall_analyzer import HIP_UP, falling, make_track, standing

FPS = 25.0


def record(seq: int) -> FrameRecord:
    t = 100.0 + seq / FPS
    return FrameRecord(
        "cam1",
        seq,
        seq * 3600,
        Fraction(1, 90000),
        int(t * 1e9),
        datetime(2026, 1, 1, tzinfo=UTC),
        False,
        100,
        TimestampQuality.OK,
    )


class ScriptedPipeline:
    """Returns the scripted tracks of each frame; records what it was given."""

    def __init__(self, tracks: list[list[Track]]) -> None:
        self.tracks = tracks
        self.shapes: list[tuple[int, ...]] = []

    def process(self, image: Any, rec: FrameRecord) -> FrameResult:
        self.shapes.append(image.shape)
        return FrameResult(rec, self.tracks[rec.seq], analysed=True)


def fall_at_full_rate() -> list[list[Track]]:
    frames = [[standing()] for _ in range(50)]  # 2 s upright
    frames += [[falling(0, k / 10)] for k in range(1, 11)]  # 0.4 s descent
    frames += [[falling(0, 1.0)] for _ in range(50)]  # 2 s lying
    return frames


def frames(count: int, shape: tuple[int, int, int] = (72, 128, 3)) -> list[tuple[FrameRecord, Any]]:
    return [(record(i), np.zeros(shape, np.uint8)) for i in range(count)]


def confirmer(pipeline: ScriptedPipeline, **kw: Any) -> FallConfirmer:
    return FallConfirmer(lambda: pipeline, lambda: FallAnalyzer("cam1"), max_width_px=1280, **kw)


def test_a_fall_seen_at_full_rate_is_confirmed() -> None:
    tracks = fall_at_full_rate()
    pipeline = ScriptedPipeline(tracks)
    events = confirmer(pipeline).confirm(frames(len(tracks)))
    assert len(events) == 1
    event = events[0]
    assert event.category is EventCategory.POSSIBLE_FALL and not event.near_miss
    assert event.evidence["reanalysed_at_full_rate"] is True
    assert len(pipeline.shapes) == len(tracks)  # every frame analysed


def test_a_stumble_is_not_confirmed() -> None:
    tracks = [[standing()] for _ in range(50)]
    tracks += [[make_track(0, HIP_UP + 30.0, 60.0, 0.6)] for _ in range(10)]
    tracks += [[standing()] for _ in range(50)]
    assert confirmer(ScriptedPipeline(tracks)).confirm(frames(len(tracks))) == []


def test_the_rate_can_be_capped() -> None:
    tracks = fall_at_full_rate()
    pipeline = ScriptedPipeline(tracks)
    confirmer(pipeline, max_fps=10.0).confirm(frames(len(tracks)))
    duration_s = len(tracks) / FPS
    assert len(pipeline.shapes) <= 10 * duration_s + 1


def test_frames_are_resized_and_rotated_like_the_live_analysis() -> None:
    pipeline = ScriptedPipeline([[] for _ in range(3)])
    image = np.zeros((100, 400, 3), np.uint8)
    image[0, 0] = 255  # top-left corner
    seen: list[Any] = []

    class Capturing(ScriptedPipeline):
        def process(self, image: Any, rec: FrameRecord) -> FrameResult:
            seen.append(image)
            return super().process(image, rec)

    capturing = Capturing(pipeline.tracks)
    FallConfirmer(
        lambda: capturing, lambda: FallAnalyzer("cam1"), max_width_px=200, rotate_deg=180
    ).confirm([(record(0), image)])
    assert seen[0].shape == (50, 200, 3)
    assert seen[0][-1, -1].max() > 0  # the corner is now bottom-right


def test_pending_confirmations_wait_for_the_frames_after_the_candidate() -> None:
    pending = PendingConfirmations(after_s=3.0)
    near_miss = make_event(
        camera_id="cam1",
        category=EventCategory.POSSIBLE_FALL,
        confidence=0.3,
        start_s=10.0,
        end_s=11.0,
        track_id=1,
        identity="stable",
        evidence={"near_fall": True},
        analysed_seqs=[],
        threshold=0.5,
    )
    assert pending.add(near_miss)
    assert pending.due(13.9) == []
    assert pending.due(14.0) == [near_miss]
    assert pending.due(20.0) == []


@pytest.mark.parametrize(
    ("category", "near_miss", "evidence"),
    [
        (EventCategory.POSSIBLE_FALL, False, {}),  # already an alert
        (EventCategory.PERSON_DOWN, True, {}),  # not a fall candidate
        (EventCategory.POSSIBLE_FALL, True, {"safe_zone": "bed"}),  # lying down in bed
    ],
)
def test_only_fall_near_misses_outside_safe_zones_are_candidates(
    category: EventCategory, near_miss: bool, evidence: dict[str, Any]
) -> None:
    event = make_event(
        camera_id="cam1",
        category=category,
        confidence=0.3 if near_miss else 0.8,
        start_s=10.0,
        end_s=11.0,
        track_id=1,
        identity="stable",
        evidence=evidence,
        analysed_seqs=[],
        threshold=0.5,
    )
    assert not PendingConfirmations(after_s=3.0).add(event)


def test_a_pipeline_sharing_the_models_analyses_every_frame() -> None:
    from fall_impact_monitor.api.perception import PerceptionPipeline
    from fall_impact_monitor.perception.cadence import CadencePolicy, RateMonitor
    from fall_impact_monitor.perception.tracking import PersonTracker
    from tests.api.test_build import FlatPose, OnePerson

    detector, pose = OnePerson(), FlatPose()
    live = PerceptionPipeline(
        detector, pose, PersonTracker(), CadencePolicy(), RateMonitor(min_fps=0.5, started_s=0.0)
    )
    full = live.sharing_models(PersonTracker(min_age_frames=1))
    image = np.zeros((72, 128, 3), np.uint8)
    analysed = [full.process(image, record(i)).analysed for i in range(25)]
    assert all(analysed)  # 25 frames in one second, every one analysed
    assert full is not live
