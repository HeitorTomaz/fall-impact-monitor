import pytest

from fall_impact_monitor.evaluation.scoring import Detection, VideoResult, score
from fall_impact_monitor.evaluation.truth import TruthEvent


def alert(start_s: float, emitted_s: float | None = None) -> Detection:
    return Detection("possible_fall", start_s, emitted_s if emitted_s is not None else start_s + 1)


def test_scoring_counts_detected_missed_duplicate_and_false_alerts() -> None:
    results = [
        # Detected once, with a second alert for the same fall (a duplicate, not false).
        VideoResult("fall1", 10.0, (TruthEvent(3.0, 6.0),), (alert(3.5, 5.0), alert(5.5))),
        # Missed: the only alert is far from the fall.
        VideoResult("fall2", 10.0, (TruthEvent(2.0, 3.0),), (alert(9.0),)),
        # Activity of daily living with one false alert.
        VideoResult("adl1", 1780.0, (), (alert(4.0),)),
    ]
    report = score(results, dataset="synthetic")
    assert report.falls == 2 and report.detected == 1
    assert report.recall == pytest.approx(0.5)
    assert report.duplicates == 1
    assert report.false_alerts == 2  # fall2's far alert and adl1's
    assert report.hours == pytest.approx(0.5)
    assert report.false_alerts_per_hour == pytest.approx(4.0)
    assert report.latency_median_s == pytest.approx(2.0)  # emitted 5.0 for a fall at 3.0
    assert report.missed == ["fall2"] and report.false == ["adl1", "fall2"]


def test_only_fall_alerts_count() -> None:
    other = Detection("possible_head_impact", 3.5, 4.0)
    results = [VideoResult("fall1", 10.0, (TruthEvent(3.0, 6.0),), (other,))]
    report = score(results, dataset="synthetic")
    assert report.detected == 0 and report.false_alerts == 0


def test_a_person_down_alert_detects_a_fall() -> None:
    down = Detection("person_down", 4.0, 20.0)
    results = [VideoResult("fall1", 60.0, (TruthEvent(3.0, 6.0),), (down,))]
    assert score(results, dataset="synthetic").detected == 1


def test_the_reported_instant_must_be_near_the_start_of_the_fall() -> None:
    """The alert shows the instant at which the fall was identified: from 2 s before to
    5 s after the labelled start, even when the labelled fall (lying) lasts longer."""
    fall = (TruthEvent(10.0, 40.0),)
    early, ok, late = alert(7.9), alert(14.9), alert(15.1)
    assert score([VideoResult("f", 60.0, fall, (early,))], dataset="x").detected == 0
    assert score([VideoResult("f", 60.0, fall, (ok,))], dataset="x").detected == 1
    report = score([VideoResult("f", 60.0, fall, (late,))], dataset="x")
    assert report.detected == 0
    assert report.false_alerts == 0 and report.duplicates == 1  # still about this fall


def test_an_alert_sent_after_the_deadline_is_only_counted_within_sixty_seconds() -> None:
    fall = (TruthEvent(10.0, 12.0),)
    sent_45 = Detection("possible_fall", 11.0, 55.0)
    report = score([VideoResult("f", 60.0, fall, (sent_45,))], dataset="x")
    assert report.detected == 0 and report.detected_within_late_deadline == 1
    sent_25 = Detection("possible_fall", 11.0, 35.0)
    report = score([VideoResult("f", 60.0, fall, (sent_25,))], dataset="x")
    assert report.detected == 1 and report.detected_within_late_deadline == 1
    assert report.deadline_s == 30.0 and report.late_deadline_s == 60.0


def test_alerts_raised_on_padding_are_counted_apart() -> None:
    """Evaluation pads clip ends with the last frame (``fim analyze --pad-end-s``)."""
    fall = (TruthEvent(8.0, 9.5),)
    padded_hit = Detection("possible_fall", 8.5, 12.0)  # sent after the 10 s clip ended
    padded_false = Detection("person_down", 1.0, 14.0)
    report = score([VideoResult("f", 10.0, fall, (padded_hit, padded_false))], dataset="x")
    assert report.detected == 1 and report.detected_on_padding == 1
    assert report.false_alerts == 1 and report.false_alerts_on_padding == 1


def test_the_report_is_deterministic() -> None:
    results = [
        VideoResult("b", 5.0, (), (alert(1.0),)),
        VideoResult("a", 5.0, (TruthEvent(1.0, 2.0),), ()),
    ]
    assert score(results, dataset="x") == score(list(reversed(results)), dataset="x")


def test_two_falls_with_overlapping_windows_each_take_their_own_alert() -> None:
    falls = (TruthEvent(2.0, 4.0), TruthEvent(5.5, 8.0))
    results = [VideoResult("two", 10.0, falls, (alert(3.0), alert(6.0)))]
    report = score(results, dataset="synthetic")
    assert report.detected == 2 and report.duplicates == 0 and report.false_alerts == 0


def test_latency_uses_the_alert_raised_first() -> None:
    late = Detection("person_down", 3.1, 25.0)
    early = Detection("possible_fall", 3.3, 4.0)
    results = [VideoResult("fall", 60.0, (TruthEvent(3.0, 6.0),), (late, early))]
    report = score(results, dataset="synthetic")
    assert report.latency_median_s == pytest.approx(1.0) and report.duplicates == 1


def test_a_subset_without_falls_has_no_recall() -> None:
    report = score([VideoResult("adl", 10.0, (), ())], dataset="synthetic")
    assert report.recall is None and report.latency_median_s is None
