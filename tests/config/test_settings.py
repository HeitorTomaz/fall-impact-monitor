from pathlib import Path

import pytest

from fall_impact_monitor.config import CameraSettings, Settings, load_cameras


def test_defaults() -> None:
    settings = Settings(_env_file=None)
    assert settings.log_level == "INFO"
    assert settings.capture_stale_timeout_s == 10.0
    assert settings.capture_packet_buffer_s == 20.0
    assert settings.mediamtx_playback_url == "http://mediamtx:9996"


def test_settings_read_prefixed_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FIM_CAPTURE_STALE_TIMEOUT_S", "4.5")
    assert Settings(_env_file=None).capture_stale_timeout_s == 4.5


def test_load_cameras_reads_declared_cameras() -> None:
    env = {
        "FIM_CAMERAS": "cam1, hall_2",
        "FIM_CAMERA_CAM1_MAIN_URL": "rtsp://h/1",
        "FIM_CAMERA_CAM1_SUB_URL": "rtsp://h/2",
        "FIM_CAMERA_HALL_2_MAIN_URL": "rtsp://h/3",
    }
    assert load_cameras(env) == [
        CameraSettings(id="cam1", main_url="rtsp://h/1", sub_url="rtsp://h/2"),
        CameraSettings(id="hall_2", main_url="rtsp://h/3", sub_url=None),
    ]


def test_no_cameras_declared_returns_empty_list() -> None:
    assert load_cameras({}) == []


def test_missing_main_url_names_the_variable_not_secrets() -> None:
    env = {"FIM_CAMERAS": "cam1", "FIM_CAMERA_CAM1_SUB_URL": "rtsp://u:secret@h/2"}
    with pytest.raises(ValueError, match="FIM_CAMERA_CAM1_MAIN_URL") as excinfo:
        load_cameras(env)
    assert "secret" not in str(excinfo.value)


@pytest.mark.parametrize("declared", ["Cam1", "cam-1", "cam 1", "cam1,,cam2"])
def test_invalid_camera_id_is_rejected(declared: str) -> None:
    with pytest.raises(ValueError, match="camera id"):
        load_cameras({"FIM_CAMERAS": declared})


def test_perception_defaults() -> None:
    settings = Settings(_env_file=None)
    assert settings.person_detector_model == "yolox_m_humanart"
    assert settings.pose_estimator_model == "rtmpose_s"
    assert settings.models_dir == "models"
    assert settings.analysis_empty_scene_fps == 3.0
    assert settings.analysis_person_visible_fps == 8.0
    assert settings.analysis_min_fps == 0.5


def test_person_detector_min_score_default_keeps_weak_detections() -> None:
    assert Settings(_env_file=None).person_detector_min_score == 0.1


def test_detection_rule_defaults() -> None:
    settings = Settings(_env_file=None)
    assert settings.alert_min_confidence == 0.5
    assert settings.fall_detector_enabled is True
    assert settings.person_down_detector_enabled is True
    assert settings.person_down_detector_after_fall_s == 30.0
    assert settings.person_down_detector_without_fall_s == 60.0


@pytest.mark.parametrize("value", ["-0.1", "1.5"])
def test_alert_min_confidence_outside_0_1_fails_fast(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv("FIM_ALERT_MIN_CONFIDENCE", value)
    with pytest.raises(ValueError, match="alert_min_confidence"):
        Settings(_env_file=None)


def test_camera_zones_file_is_read() -> None:
    env = {
        "FIM_CAMERAS": "cam1",
        "FIM_CAMERA_CAM1_MAIN_URL": "rtsp://h/1",
        "FIM_CAMERA_CAM1_ZONES_FILE": "local/zones/cam1.json",
    }
    assert load_cameras(env)[0].zones_file == "local/zones/cam1.json"


def test_recording_defaults() -> None:
    settings = Settings(_env_file=None)
    assert settings.recordings_dir == "/recordings"
    assert settings.event_clip_dir == "/events"
    assert settings.recordings_max_age_h == 168.0
    assert settings.recordings_max_gb == 200.0
    assert settings.recordings_disk_min_free_gb == 10.0
    assert settings.event_clip_pre_s == 10.0  # a fixed 30 s window around the event start
    assert settings.event_clip_post_s == 20.0
    assert settings.event_clip_max_mb == 50.0
    assert settings.event_clip_retention_days == 30.0


def test_recordings_cap_must_be_positive(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FIM_RECORDINGS_MAX_GB", "0")
    with pytest.raises(ValueError, match="recordings_max_gb"):
        Settings(_env_file=None)


def test_database_url_quotes_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FIM_DB_PASSWORD", "p@ss/word")
    url = Settings(_env_file=None).database_url()
    assert url == "postgresql+asyncpg://fim:p%40ss%2Fword@postgres:5432/fim"


def test_database_url_without_password_names_the_variable() -> None:
    with pytest.raises(ValueError, match="FIM_DB_PASSWORD"):
        Settings(_env_file=None).database_url()


@pytest.mark.parametrize(
    ("declared", "message"),
    [("cam1, cam1", "declared twice"), ("cam1, cam1sub", "substream path")],
)
def test_camera_ids_must_not_clash_with_each_other_or_a_substream(
    declared: str, message: str
) -> None:
    env = {
        "FIM_CAMERAS": declared,
        "FIM_CAMERA_CAM1_MAIN_URL": "rtsp://h/1",
        "FIM_CAMERA_CAM1_SUB_URL": "rtsp://h/2",
        "FIM_CAMERA_CAM1SUB_MAIN_URL": "rtsp://h/3",
    }
    with pytest.raises(ValueError, match=message):
        load_cameras(env)


def test_tests_never_read_a_local_env_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # A developer's .env (real Telegram token) must never reach a test.
    (tmp_path / ".env").write_text("FIM_TELEGRAM_ENABLED=true\nFIM_ALERT_LANGUAGE=pt-BR\n")
    monkeypatch.chdir(tmp_path)
    settings = Settings()
    assert settings.telegram_enabled is False
    assert settings.alert_language == "en"


def test_a_camera_can_be_rotated_upside_down() -> None:
    env = {
        "FIM_CAMERAS": "cam1",
        "FIM_CAMERA_CAM1_MAIN_URL": "rtsp://h/1",
        "FIM_CAMERA_CAM1_ROTATE_DEG": "180",
    }
    assert load_cameras(env)[0].rotate_deg == 180
    assert load_cameras({**env, "FIM_CAMERA_CAM1_ROTATE_DEG": ""})[0].rotate_deg == 0
    env.pop("FIM_CAMERA_CAM1_ROTATE_DEG")
    assert load_cameras(env)[0].rotate_deg == 0  # a camera standing upright by default


@pytest.mark.parametrize("value", ["90", "-180", "half"])
def test_an_unsupported_rotation_names_the_variable(value: str) -> None:
    env = {
        "FIM_CAMERAS": "cam1",
        "FIM_CAMERA_CAM1_MAIN_URL": "rtsp://h/1",
        "FIM_CAMERA_CAM1_ROTATE_DEG": value,
    }
    with pytest.raises(ValueError, match="FIM_CAMERA_CAM1_ROTATE_DEG"):
        load_cameras(env)


def test_video_uploads_get_a_long_timeout_by_default() -> None:
    # A 50 MB clip needs minutes on a home uplink; aiogram's default is 60 s.
    assert Settings().telegram_video_upload_timeout_s == 600.0
    with pytest.raises(ValueError):
        Settings(telegram_video_upload_timeout_s=0)


def test_full_rate_re_analysis_is_capped_by_default() -> None:
    # Every frame of a 6 s window costs about 15-19 s of four cores; 12 fps halves it.
    assert Settings().fall_confirmation_max_fps == 12.0


def test_person_detector_rate_is_capped_by_default() -> None:
    # The detector is most of the cost of an analysed frame (#68); pose follows people.
    assert Settings().person_detector_max_fps == 4.0
    assert Settings(person_detector_max_fps=0).person_detector_max_fps == 0.0
    with pytest.raises(ValueError):
        Settings(person_detector_max_fps=-1)
