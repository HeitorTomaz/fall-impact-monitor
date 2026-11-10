import os
import subprocess
import sys

import fall_impact_monitor


def test_version_is_exposed() -> None:
    assert fall_impact_monitor.__version__ == "0.1.0"


def _telemetry_setting_after_import(preset: str | None) -> str:
    env = {k: v for k, v in os.environ.items() if k != "ORT_DISABLE_TELEMETRY"}
    if preset is not None:
        env["ORT_DISABLE_TELEMETRY"] = preset
    code = "import os, fall_impact_monitor; print(os.environ.get('ORT_DISABLE_TELEMETRY'))"
    out = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True
    )
    return out.stdout.strip()


def test_importing_the_package_turns_onnxruntime_telemetry_off() -> None:
    assert _telemetry_setting_after_import(None) == "1"


def test_an_explicit_onnxruntime_telemetry_setting_is_kept() -> None:
    assert _telemetry_setting_after_import("0") == "0"
