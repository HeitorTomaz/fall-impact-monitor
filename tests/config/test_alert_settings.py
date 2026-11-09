import json

import pytest

from fall_impact_monitor.config import CameraSettings, Settings
from fall_impact_monitor.config.snapshot import config_hash, config_snapshot


def test_alert_defaults() -> None:
    settings = Settings(_env_file=None)
    assert settings.alert_language == "en"
    assert settings.alert_timezone == "UTC"
    assert settings.alert_aggregation_window_s == 300.0
    assert settings.alert_late_after_s == 60.0
    assert settings.event_log_file == "/events/events.jsonl"
    assert settings.outbox_dir == "/events/outbox"
    assert settings.telegram_enabled is False
    assert settings.telegram_chat_id_list() == []
    assert settings.webhook_url is None
    assert settings.webhook_timeout_s == 10.0


def test_portuguese_and_a_local_timezone_are_accepted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FIM_ALERT_LANGUAGE", "pt-BR")
    monkeypatch.setenv("FIM_ALERT_TIMEZONE", "America/Sao_Paulo")
    settings = Settings(_env_file=None)
    assert (settings.alert_language, settings.alert_timezone) == ("pt-BR", "America/Sao_Paulo")


def test_an_unknown_timezone_fails_fast(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FIM_ALERT_TIMEZONE", "Mars/Olympus")
    with pytest.raises(ValueError, match="alert_timezone"):
        Settings(_env_file=None)


def test_telegram_needs_a_token_and_chats_and_never_shows_them(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FIM_TELEGRAM_ENABLED", "true")
    with pytest.raises(ValueError, match="FIM_TELEGRAM_BOT_TOKEN"):
        Settings(_env_file=None)
    monkeypatch.setenv("FIM_TELEGRAM_BOT_TOKEN", "123:secret-token")
    with pytest.raises(ValueError, match="FIM_TELEGRAM_CHAT_IDS"):
        Settings(_env_file=None)
    monkeypatch.setenv("FIM_TELEGRAM_CHAT_IDS", "-1001, 42")
    settings = Settings(_env_file=None)
    assert settings.telegram_chat_id_list() == [-1001, 42]
    assert "secret-token" not in repr(settings)


def test_the_configuration_snapshot_masks_secrets_and_has_a_stable_hash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FIM_DB_PASSWORD", "db-secret")
    monkeypatch.setenv("FIM_WEBHOOK_URL", "https://hooks.example/secret-path")
    cameras = [CameraSettings("cam1", "rtsp://user:cam-secret@host/1", None)]
    snapshot = config_snapshot(Settings(_env_file=None), cameras)
    text = json.dumps(snapshot)
    assert "db-secret" not in text and "secret-path" not in text and "cam-secret" not in text
    assert snapshot["settings"]["alert_min_confidence"] == 0.5
    assert snapshot["cameras"][0]["id"] == "cam1"
    first = config_hash(snapshot)
    assert first == config_hash(json.loads(text)) and len(first) == 16
    snapshot["settings"]["alert_min_confidence"] = 0.6
    assert config_hash(snapshot) != first


def test_start_up_errors_never_show_secret_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FIM_TELEGRAM_ENABLED", "true")
    monkeypatch.setenv("FIM_TELEGRAM_BOT_TOKEN", "123:very-secret-token")
    monkeypatch.setenv("FIM_DB_PASSWORD", "db-secret-value")
    with pytest.raises(ValueError) as info:
        Settings(_env_file=None)
    assert "very-secret-token" not in str(info.value)
    assert "db-secret-value" not in str(info.value)
