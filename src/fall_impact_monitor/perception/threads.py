"""ONNX Runtime threads per session.

By default a session starts one thread per core and keeps them spinning between runs.
With several cameras, each with its own detector and pose sessions, those threads fight
for the same cores: on 4 CPUs, four cameras ran about 4.5 times slower than with one
thread per session (#68). The cameras now share the CPUs instead."""

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import onnxruntime as ort


def available_cpus() -> int:
    """CPUs this process may run on. A container limited by a CPU quota (``--cpus``) still
    sees every core: set ``FIM_INFERENCE_THREADS`` there."""
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:  # not on Linux
        return os.cpu_count() or 1


def inference_threads(setting: int, *, cameras: int, cpus: int) -> int:
    """Threads per session: ``setting`` when positive, otherwise the CPUs shared by the
    cameras (at least one each)."""
    if setting > 0:
        return setting
    return max(1, cpus // max(1, cameras))


def session_options(threads: int) -> "ort.SessionOptions":
    """``threads`` for each operator, one operator at a time, and no spinning while idle."""
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.add_session_config_entry("session.intra_op.allow_spinning", "0")
    return options
