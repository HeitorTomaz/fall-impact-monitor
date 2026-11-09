"""Every setting is documented, and names follow FIM_<PROCESS>_<WHAT>_<UNIT> (CLAUDE.md §11)."""

import re
from pathlib import Path

import pytest

from fall_impact_monitor.config import Settings

ROOT = Path(__file__).resolve().parents[2]
DOCUMENTS = ("README.md", ".env.example")

# Renamed on 2026-11-09 without aliases; they must not come back. Accepted ADRs keep
# them as history and are not checked.
OLD_NAMES = (
    "FIM_STALE_TIMEOUT_S",
    "FIM_PACKET_BUFFER_S",
    "FIM_DETECTOR_MODEL",
    "FIM_DETECTOR_SCORE_THRESHOLD",
    "FIM_NEW_TRACK_MIN_SCORE",
    "FIM_TRACK_LOST_S",
    "FIM_POSE_MODEL",
    "FIM_EMPTY_FPS",
    "FIM_PERSON_FPS",
    "FIM_PERSON_HOLD_S",
    "FIM_MIN_ANALYSIS_FPS",
    "FIM_FLOOR_DETECTOR_ENABLED",
    "FIM_PERSON_DOWN_S",
    "FIM_FLOOR_NO_FALL_S",
    "FIM_ALERT_CONFIDENCE_THRESHOLD",
    "FIM_MEDIAMTX_API_URL",
    "FIM_MODELS",
    "SIM_MEDIA_DIR",
    "SIM_FILE",
    "SIM_TARGET",
)
CHECKED = (
    "src",
    "tests",
    "docs",
    "deploy",
    "README.md",
    ".env.example",
    "compose.yaml",
    "Dockerfile",
    ".github",
    "CLAUDE.md",
    "MODELS.md",
)


def setting_names() -> list[str]:
    return [f"FIM_{name.upper()}" for name in Settings.model_fields]


@pytest.mark.parametrize("document", DOCUMENTS)
def test_every_setting_is_documented(document: str) -> None:
    text = (ROOT / document).read_text(encoding="utf-8")
    assert [n for n in setting_names() if not re.search(rf"\b{n}\b", text)] == []


def checked_files() -> list[Path]:
    files: list[Path] = []
    for entry in CHECKED:
        path = ROOT / entry
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            files += [
                p
                for p in path.rglob("*")
                if p.is_file()
                and p.suffix in {".py", ".md", ".yaml", ".yml", ".sh", ".toml", ".json", ""}
                and "adr" not in p.parts
                and "__pycache__" not in p.parts
                and p.name != "test_documentation.py"
            ]
    return files


def test_old_variable_names_are_gone() -> None:
    pattern = re.compile(r"(?<![A-Z0-9_])(" + "|".join(OLD_NAMES) + r")(?![A-Z0-9_])")
    found = {
        f"{path.relative_to(ROOT)}: {match}"
        for path in checked_files()
        for match in pattern.findall(path.read_text(encoding="utf-8", errors="ignore"))
    }
    assert sorted(found) == []
