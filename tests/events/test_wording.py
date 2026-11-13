from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from fall_impact_monitor.events import EventCategory, make_event
from fall_impact_monitor.events.record import EventRecord
from fall_impact_monitor.events.wording import (
    alert_text,
    button_labels,
    delivery_test_text,
    message,
    video_caption,
)

T0 = datetime(2026, 11, 9, 23, 30, 5, tzinfo=UTC)
LANGUAGES = ("en", "pt-BR")


def record(
    category: EventCategory, evidence: dict[str, Any] | None = None, **kw: Any
) -> EventRecord:
    event = make_event(
        camera_id="cam1",
        category=category,
        confidence=kw.get("confidence", 0.724),
        start_s=0.0,
        end_s=1.0,
        track_id=1,
        identity="stable",
        evidence=evidence or {},
        analysed_seqs=[],
        threshold=0.5,
    )
    return EventRecord(event, T0, T0 + timedelta(seconds=1), T0 + timedelta(seconds=2), "h")


CASES = [
    (EventCategory.POSSIBLE_FALL, {"max_drop_heights": 0.61, "recovered": True}),
    (EventCategory.PERSON_DOWN, {"fall_observed": True, "lying_s": 31.2}),
    (EventCategory.PERSON_DOWN, {"fall_observed": False, "lying_s": 64.0}),
    (EventCategory.POSSIBLE_HEAD_IMPACT, {}),
    (EventCategory.MONITORING_DEGRADED, {"reason": "camera offline"}),
    (EventCategory.MONITORING_RESTORED, {}),
]


@pytest.mark.parametrize("language", LANGUAGES)
@pytest.mark.parametrize(("category", "evidence"), CASES)
def test_every_category_is_worded_honestly(
    language: str, category: EventCategory, evidence: dict[str, Any]
) -> None:
    text = alert_text(
        record(category, evidence),
        language=language,  # type: ignore[arg-type]
        timezone="UTC",
        now_utc=T0 + timedelta(seconds=5),
        late_after_s=60,
    )
    assert "cam1" in text
    lowered = text.lower()
    assert "confirmed" not in lowered and "confirmad" not in lowered
    assert "injur" not in lowered and "lesão" not in lowered


def test_english_fall_alert() -> None:
    text = alert_text(
        record(EventCategory.POSSIBLE_FALL, {"max_drop_heights": 0.61, "recovered": True}),
        language="en",
        timezone="America/Sao_Paulo",
        now_utc=T0 + timedelta(seconds=5),
        late_after_s=60,
    )
    assert text.startswith("⚠️ Possible fall")
    assert "72% (rule-based, uncalibrated)" in text
    assert "2026-11-09 20:30:05" in text  # UTC-3
    assert "does not replace supervision" in text
    assert "Late" not in text


def test_portuguese_person_down_without_fall() -> None:
    text = alert_text(
        record(EventCategory.PERSON_DOWN, {"fall_observed": False, "lying_s": 64.0}),
        language="pt-BR",
        timezone="UTC",
        now_utc=T0,
        late_after_s=60,
    )
    assert text.startswith("⚠️ Pessoa deitada no chão há 64 s (queda não observada)")
    assert "72% (por regras, não calibrada)" in text
    assert "Não substitui a supervisão" in text


def test_a_late_alert_says_how_late_and_repeats_are_counted() -> None:
    text = alert_text(
        record(EventCategory.POSSIBLE_FALL),
        language="en",
        timezone="UTC",
        now_utc=T0 + timedelta(seconds=1 + 75),
        late_after_s=60,
        repeats=2,
    )
    assert "Late: sent 75 s after the event" in text
    assert "Repeated 2 times" in text


@pytest.mark.parametrize("language", LANGUAGES)
def test_buttons_and_short_messages_exist_in_both_languages(language: str) -> None:
    labels = button_labels(language)  # type: ignore[arg-type]
    assert set(labels) == {"seen", "false_alarm", "confirmed", "video"}
    assert delivery_test_text(language)  # type: ignore[arg-type]
    for key in (
        "video_sending",
        "video_when_ready",
        "video_missing",
        "feedback_saved",
        "unknown_event",
    ):
        assert message(language, key)  # type: ignore[arg-type]


def test_video_caption_shows_both_windows_in_local_time() -> None:
    sidecar = {
        "requested_start": "2026-11-09T23:29:45+00:00",
        "requested_end": "2026-11-09T23:31:06+00:00",
        "actual_start": "2026-11-09T23:29:44+00:00",
        "actual_end": "2026-11-09T23:31:06+00:00",
        "gaps": [{"start": "2026-11-09T23:30:10+00:00", "end": "2026-11-09T23:30:20+00:00"}],
        "truncated": False,
    }
    caption = video_caption(sidecar, language="en", timezone="America/Sao_Paulo")
    assert "Requested 20:29:45 - 20:31:06" in caption
    assert "Recorded 20:29:44 - 20:31:06" in caption
    assert "1 gap" in caption
