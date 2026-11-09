import warnings

import numpy as np

from fall_impact_monitor.perception.detector import Detection
from fall_impact_monitor.perception.pose import Pose
from fall_impact_monitor.perception.tracking import PersonTracker, Track

STEP_S = 0.2


def walker(x: float, y: float = 10.0) -> Detection:
    return Detection(box_xyxy=(x, y, x + 40.0, y + 100.0), score=0.9)


def run(
    tracker: PersonTracker,
    frames: list[list[Detection]],
    start_s: float = 0.0,
    times: list[float] | None = None,
) -> list[list[Track]]:
    out = []
    for i, detections in enumerate(frames):
        t = times[i] if times else start_s + i * STEP_S
        out.append(tracker.update(detections, [None] * len(detections), t))
    return out


def test_separated_walkers_keep_their_ids_with_irregular_timestamps() -> None:
    frames = [[walker(10 + 2 * i), walker(300 - 2 * i)] for i in range(30)]
    times = [0.0, 0.2, 0.2, 0.5, 0.5, 0.6] + [0.8 + 0.2 * i for i in range(24)]
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # the tracker wrapper must not leak warnings
        results = run(PersonTracker(), frames, times=times)
    ids_late = {t.track_id for t in results[-1]}
    assert len(ids_late) == 2
    for tracks in results[5:]:
        assert {t.track_id for t in tracks} == ids_late


def test_tracks_carry_box_score_and_pose() -> None:
    pose = Pose(keypoints=np.zeros((17, 2), np.float32), scores=np.ones(17, np.float32))
    tracker = PersonTracker()
    tracks: list[Track] = []
    for i in range(5):
        tracks = tracker.update([walker(10 + i)], [pose], i * STEP_S)
    (track,) = tracks
    assert track.box_xyxy == (14.0, 10.0, 54.0, 110.0)
    assert track.score == 0.9
    assert track.keypoints is not None and track.keypoints.shape == (17, 2)


def test_crossing_walkers_are_uncertain_during_and_after_overlap() -> None:
    tracker = PersonTracker(uncertain_hold_s=1.0, min_age_frames=3)
    positions = [(10 + 20 * i, 300 - 20 * i) for i in range(16)]  # they cross around i = 7
    results = run(tracker, [[walker(a), walker(b)] for a, b in positions])
    before = results[4]
    assert {t.identity for t in before} == {"stable"}
    crossing = results[7]
    assert crossing and {t.identity for t in crossing} == {"uncertain"}
    just_after = results[9]  # 0.4 s after the overlap ended, within the hold time
    assert {t.identity for t in just_after} == {"uncertain"}


def test_new_track_is_uncertain_until_old_enough() -> None:
    tracker = PersonTracker(min_age_frames=4)
    results = run(tracker, [[walker(10 + i)] for i in range(8)])
    identities = [tracks[0].identity for tracks in results if tracks]
    assert identities[0] == "uncertain"
    assert identities[-1] == "stable"


def test_frames_without_people_return_no_tracks() -> None:
    tracker = PersonTracker()
    assert tracker.update([], [], 0.0) == []
    run(tracker, [[walker(10)]] * 4)
    assert tracker.update([], [], 2.0) == []


def weak(x: float, score: float) -> Detection:
    return Detection(box_xyxy=(x, 10.0, x + 40.0, 110.0), score=score)


def test_moderately_confident_walker_is_tracked() -> None:
    tracker = PersonTracker(new_track_min_score=0.3)
    results = run(tracker, [[weak(10 + i, 0.5)] for i in range(10)])
    assert results[-1] and results[-1][0].track_id == results[3][0].track_id


def test_one_missed_detection_keeps_the_track_id() -> None:
    tracker = PersonTracker(track_lost_s=3.0)
    frames = [[walker(10 + i)] for i in range(8)] + [[]] * 3 + [[walker(20 + i)] for i in range(5)]
    results = run(tracker, frames)
    assert results[7] and results[-1]
    assert results[7][0].track_id == results[-1][0].track_id


def test_overlap_with_an_unconfirmed_person_makes_identity_uncertain() -> None:
    tracker = PersonTracker(min_age_frames=1)
    results = run(tracker, [[walker(10)] for _ in range(6)])
    assert results[-1][0].identity == "stable"
    newcomer = weak(25, 0.2)  # weak, overlapping, never a confirmed track
    (track,) = tracker.update([walker(10), newcomer], [None, None], 6 * STEP_S)
    assert track.identity == "uncertain"


def test_tracks_report_their_detection_index() -> None:
    tracker = PersonTracker()
    tracks = run(tracker, [[weak(400, 0.05), walker(10 + i)] for i in range(6)])[-1]
    assert [t.detection_index for t in tracks] == [1]
