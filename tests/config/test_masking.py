from fall_impact_monitor.config import mask_url


def test_credentials_are_masked() -> None:
    assert (
        mask_url("rtsp://user:secret@cam.local:554/stream1") == "rtsp://***@cam.local:554/stream1"
    )


def test_username_only_is_masked() -> None:
    assert mask_url("rtsp://user@cam.local/stream1") == "rtsp://***@cam.local/stream1"


def test_url_without_credentials_is_unchanged() -> None:
    assert mask_url("rtsp://cam.local:554/stream1") == "rtsp://cam.local:554/stream1"


def test_local_path_is_unchanged() -> None:
    assert mask_url("/media/video.mkv") == "/media/video.mkv"


def test_credentials_inside_free_text_are_masked() -> None:
    text = "failed to open rtsp://u:p@10.0.0.2/s: timeout"
    assert mask_url(text) == "failed to open rtsp://***@10.0.0.2/s: timeout"
