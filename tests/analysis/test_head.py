import math
from itertools import pairwise

import numpy as np

from fall_impact_monitor.analysis.head import HeadConfig, HeadTracker, head_point, head_stop
from fall_impact_monitor.analysis.one_euro import OneEuroFilter
from fall_impact_monitor.analysis.zones import Zone
from fall_impact_monitor.perception.tracking import Track

SIZE = (1000, 1000)
WALL = Zone("wall", "hazard", ((0.8, 0.0), (1.0, 0.0), (1.0, 1.0), (0.8, 1.0)))
HEIGHT = 400.0


def test_one_euro_keeps_a_constant_and_follows_a_step() -> None:
    f = OneEuroFilter(min_cutoff=1.0, beta=0.0)
    assert [f.filter(t * 0.1, 5.0) for t in range(5)] == [5.0] * 5
    values = [f.filter(0.5 + t * 0.1, 10.0) for t in range(30)]
    assert 5.0 < values[0] < 10.0 and math.isclose(values[-1], 10.0, abs_tol=0.05)
    assert f.filter(3.0, 10.0) == f.filter(3.0, 50.0)  # zero time step: unchanged


def track(nose: tuple[float, float] | None, box_top: float = 300.0) -> Track:
    points = np.zeros((17, 2), np.float32)
    scores = np.zeros(17, np.float32)
    if nose is not None:
        points[0] = nose
        scores[0] = 0.9
    box = (400.0, box_top, 500.0, box_top + HEIGHT)
    return Track(1, box, 0.9, points, scores, "stable", 10, 0)  # type: ignore[arg-type]


def test_the_head_point_is_the_head_keypoints_or_the_top_of_the_box() -> None:
    assert head_point(track((450.0, 320.0))) == (450.0, 320.0)
    assert head_point(track(None)) == (450.0, 340.0)  # top-centre, a tenth down the box


def run(xs: list[float], *, valid: list[bool] | None = None, fps: float = 15.0) -> list:  # type: ignore[type-arg]
    tracker = HeadTracker(HeadConfig())
    out = []
    for i, x in enumerate(xs):
        ok = True if valid is None else valid[i]
        out.append(tracker.update(i / fps, (x, 300.0), HEIGHT, valid_time=ok))
    return out


def test_speed_is_in_person_heights_per_second() -> None:
    samples = [s for s in run([400.0 + 40.0 * i for i in range(20)]) if s is not None]
    assert samples and math.isclose(samples[-1].speed_hps, 40.0 * 15 / HEIGHT, rel_tol=0.15)


def test_invalid_timestamps_break_the_velocity_chain() -> None:
    samples = run([400.0, 410.0, 420.0, 900.0, 430.0], valid=[True, True, True, False, True])
    assert samples[3] is None and samples[4] is None  # no velocity across the invalid frame


def test_a_fast_head_stopping_at_a_wall_is_evidence() -> None:
    xs = [500.0 + 40.0 * i for i in range(8)] + [820.0] * 4  # 1.5 heights/s, then stops
    samples = [s for s in run(xs) if s is not None]
    strengths = [head_stop(a, b, [WALL], SIZE, HeadConfig()) for a, b in pairwise(samples)]
    assert max(s for s in strengths if s is not None) > 0.5


def test_the_same_stop_away_from_hazard_zones_is_not_evidence() -> None:
    xs = [100.0 + 40.0 * i for i in range(8)] + [420.0] * 4
    samples = [s for s in run(xs) if s is not None]
    assert all(head_stop(a, b, [WALL], SIZE, HeadConfig()) is None for a, b in pairwise(samples))


def test_a_slow_approach_is_not_evidence() -> None:
    xs = [700.0 + 5.0 * i for i in range(30)]
    samples = [s for s in run(xs) if s is not None]
    assert all(head_stop(a, b, [WALL], SIZE, HeadConfig()) is None for a, b in pairwise(samples))
