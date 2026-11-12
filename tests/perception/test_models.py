import hashlib
import zipfile
from pathlib import Path

import pytest

from fall_impact_monitor.perception.models import ModelError, fetch, load_manifest, resolve


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


ONNX = b"fake-onnx-bytes"


def write_manifest(path: Path, url: str, download_sha: str, member: str | None) -> Path:
    member_line = f'member = "{member}"\n' if member else ""
    path.write_text(
        f'[models.pose_s]\nurl = "{url}"\nsha256 = "{download_sha}"\n{member_line}'
        f'onnx_sha256 = "{sha(ONNX)}"\nlicense = "Apache-2.0 (repository)"\n'
        'training_data = "body7"\n',
        encoding="utf-8",
    )
    return path


def copy_downloader(source: Path):  # type: ignore[no-untyped-def]
    def download(url: str, dest: Path) -> None:
        dest.write_bytes(source.read_bytes())

    return download


def test_load_manifest(tmp_path: Path) -> None:
    manifest = load_manifest(write_manifest(tmp_path / "m.toml", "https://x/y.onnx", "ab", None))
    spec = manifest["pose_s"]
    assert spec.url == "https://x/y.onnx" and spec.member is None
    assert spec.filename == "pose_s.onnx"


def test_fetch_plain_file_and_resolve(tmp_path: Path) -> None:
    source = tmp_path / "src.onnx"
    source.write_bytes(ONNX)
    spec = load_manifest(write_manifest(tmp_path / "m.toml", "https://x/y.onnx", sha(ONNX), None))[
        "pose_s"
    ]
    models = tmp_path / "models"
    path = fetch(spec, models, downloader=copy_downloader(source))
    assert path.read_bytes() == ONNX
    assert resolve(spec, models) == path


def test_fetch_extracts_member_from_zip(tmp_path: Path) -> None:
    archive = tmp_path / "a.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("deep/dir/end2end.onnx", ONNX)
        zf.writestr("deep/dir/other.json", b"{}")
    spec = load_manifest(
        write_manifest(
            tmp_path / "m.toml",
            "https://x/a.zip",
            sha(archive.read_bytes()),
            "deep/dir/end2end.onnx",
        )
    )["pose_s"]
    path = fetch(spec, tmp_path / "models", downloader=copy_downloader(archive))
    assert path.read_bytes() == ONNX
    assert sorted(p.name for p in (tmp_path / "models").iterdir()) == ["pose_s.onnx"]


def test_download_hash_mismatch_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "src.onnx"
    source.write_bytes(b"tampered")
    spec = load_manifest(write_manifest(tmp_path / "m.toml", "https://x/y.onnx", sha(ONNX), None))[
        "pose_s"
    ]
    with pytest.raises(ModelError, match="sha256"):
        fetch(spec, tmp_path / "models", downloader=copy_downloader(source))
    assert not (tmp_path / "models" / "pose_s.onnx").exists()


def test_resolve_names_the_missing_file(tmp_path: Path) -> None:
    spec = load_manifest(write_manifest(tmp_path / "m.toml", "https://x/y.onnx", "ab", None))[
        "pose_s"
    ]
    with pytest.raises(ModelError, match=r"pose_s.onnx"):
        resolve(spec, tmp_path / "models")


def test_resolve_rejects_a_modified_file(tmp_path: Path) -> None:
    spec = load_manifest(write_manifest(tmp_path / "m.toml", "https://x/y.onnx", "ab", None))[
        "pose_s"
    ]
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "pose_s.onnx").write_bytes(b"other")
    with pytest.raises(ModelError, match="sha256"):
        resolve(spec, tmp_path / "models")


def test_models_verify_command_reports_missing_files(tmp_path: Path) -> None:
    from fall_impact_monitor.cli.main import main

    manifest = write_manifest(tmp_path / "m.toml", "https://x/y.onnx", "ab", None)
    code = main(
        ["models", "verify", "--manifest", str(manifest), "--models-dir", str(tmp_path / "models")]
    )
    assert code == 1


def test_repository_manifest_lists_the_default_models() -> None:
    manifest = load_manifest(Path(__file__).parents[2] / "models.toml")
    assert {"yolox_s", "yolox_tiny", "rtmpose_s", "rtmpose_m"} <= manifest.keys()
    assert all(len(spec.sha256) == 64 and len(spec.onnx_sha256) == 64 for spec in manifest.values())


# The files the Dockerfile's "models" stage copies: fetching weights must not depend on the
# rest of the source, so that a source change does not download them again.
FETCH_FILES = [
    "__init__.py",
    "cli/__init__.py",
    "cli/models.py",
    "perception/__init__.py",
    "perception/models.py",
]


def test_the_models_command_runs_from_its_own_files_without_dependencies(tmp_path: Path) -> None:
    import shutil
    import subprocess
    import sys

    import fall_impact_monitor

    package = Path(fall_impact_monitor.__file__).parent
    for name in FETCH_FILES:
        target = tmp_path / "src" / "fall_impact_monitor" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(package / name, target)
    manifest = write_manifest(tmp_path / "m.toml", "https://x/y.onnx", "ab", None)
    command = ["models", "verify", "--manifest", str(manifest), "--models-dir", str(tmp_path)]
    out = subprocess.run(
        [sys.executable, "-S", "-m", "fall_impact_monitor.cli.models", *command],
        env={"PYTHONPATH": str(tmp_path / "src")},
        capture_output=True,
        text=True,
    )
    assert out.returncode == 1, out.stderr
    assert "models: " in out.stdout
