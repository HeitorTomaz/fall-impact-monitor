import numpy as np
import pytest

from fall_impact_monitor.analysis.audio import ThumpDetector

RATE = 8000
rng = np.random.default_rng(1)


def floor(seconds: float) -> np.ndarray:
    return (rng.normal(0, 0.01, int(seconds * RATE))).astype(np.float32)


def burst(at_s: float, signal: np.ndarray, *, length_s: float = 0.03) -> np.ndarray:
    out = signal.copy()
    start = int(at_s * RATE)
    out[start : start + int(length_s * RATE)] += rng.normal(0, 0.5, int(length_s * RATE)).astype(
        np.float32
    )
    return out


def detect(signal: np.ndarray, *, chunk: int | None = None, t0: float = 100.0) -> list:  # type: ignore[type-arg]
    detector = ThumpDetector(sample_rate=RATE, min_db=12.0)
    if chunk is None:
        return detector.feed(t0, signal)
    found = []
    for start in range(0, len(signal), chunk):
        found += detector.feed(t0 + start / RATE, signal[start : start + chunk])
    return found


def test_a_short_loud_burst_is_one_thump_at_its_time() -> None:
    thumps = detect(burst(1.0, floor(2.0)))
    assert len(thumps) == 1
    assert thumps[0].t_s == pytest.approx(101.0, abs=0.05)
    assert 0.0 < thumps[0].strength <= 1.0


def test_steady_sound_is_not_a_thump() -> None:
    t = np.arange(2 * RATE) / RATE
    tone = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    assert detect(tone) == []
    assert detect(floor(2.0)) == []


def test_two_bursts_close_together_are_one_thump() -> None:
    assert len(detect(burst(1.1, burst(1.0, floor(2.0))))) == 1


def test_the_result_does_not_depend_on_how_the_audio_is_chunked() -> None:
    signal = burst(1.0, floor(2.0))
    whole = [round(t.t_s, 3) for t in detect(signal.copy())]
    chunked = [round(t.t_s, 3) for t in detect(signal.copy(), chunk=800)]
    assert whole == chunked


def test_thump_times_follow_the_audio_after_a_gap() -> None:
    detector = ThumpDetector(sample_rate=RATE, min_db=12.0)
    assert detector.feed(100.0, floor(2.0)) == []
    later = detector.feed(130.0, burst(1.0, floor(2.0)))  # 28 s without audio, then a thump
    assert len(later) == 1
    assert later[0].t_s == pytest.approx(131.0, abs=0.05)
