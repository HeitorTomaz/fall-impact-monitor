import pytest

from fall_impact_monitor.config import CameraSettings, Settings, load_cameras


def test_defaults() -> None:
    settings = Settings(_env_file=None)
    assert settings.log_level == "INFO"
    assert settings.stale_timeout_s == 10.0
    assert settings.packet_buffer_s == 20.0
    assert settings.mediamtx_api_url == "http://mediamtx:9996"


def test_settings_read_prefixed_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FIM_STALE_TIMEOUT_S", "4.5")
    assert Settings(_env_file=None).stale_timeout_s == 4.5


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
