"""ONNX Runtime threads per session (#68): several cameras share the CPU instead of each
session starting a thread per core and spinning while it waits."""

from pathlib import Path

import pytest

from fall_impact_monitor.perception.threads import inference_threads, session_options


@pytest.mark.parametrize(
    ("setting", "cameras", "cpus", "expected"),
    [
        (0, 4, 8, 2),  # automatic: the CPUs shared by the cameras
        (0, 4, 4, 1),
        (0, 3, 4, 1),
        (0, 16, 4, 1),  # never below one
        (0, 1, 6, 6),  # one camera (fim analyze, fim eval) keeps every CPU
        (0, 0, 6, 6),  # no camera declared counts as one
        (3, 4, 8, 3),  # a fixed value wins
    ],
)
def test_threads_per_session(setting: int, cameras: int, cpus: int, expected: int) -> None:
    assert inference_threads(setting, cameras=cameras, cpus=cpus) == expected


def test_session_options_limit_threads_and_stop_spinning() -> None:
    options = session_options(2)
    assert options.intra_op_num_threads == 2
    assert options.inter_op_num_threads == 1
    assert options.get_session_config_entry("session.intra_op.allow_spinning") == "0"


def test_sessions_get_the_limited_options(monkeypatch: pytest.MonkeyPatch) -> None:
    import onnxruntime as ort

    from fall_impact_monitor.perception.detector import create_session

    seen: list[object] = []
    monkeypatch.setattr(
        ort,
        "InferenceSession",
        lambda path, sess_options=None, providers=None: seen.append(sess_options),
    )
    create_session(Path("model.onnx"), threads=1)
    create_session(Path("model.onnx"))
    assert seen[0].intra_op_num_threads == 1  # type: ignore[attr-defined]
    assert seen[1] is None  # ONNX Runtime's own default
