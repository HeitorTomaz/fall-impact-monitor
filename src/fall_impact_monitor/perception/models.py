"""Model manifest: where weights come from and how they are verified.

Weights are never committed. They are downloaded from their original source
(normally at image build time), verified by SHA-256, and stored as
``<models_dir>/<name>.onnx``.
"""

import hashlib
import shutil
import tempfile
import tomllib
import urllib.request
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

Downloader = Callable[[str, Path], None]


class ModelError(RuntimeError):
    """A model file is missing, corrupt or cannot be fetched."""


@dataclass(frozen=True)
class ModelSpec:
    name: str
    url: str
    sha256: str
    onnx_sha256: str
    license: str
    training_data: str
    member: str | None = None

    @property
    def filename(self) -> str:
        return f"{self.name}.onnx"


def load_manifest(path: Path) -> dict[str, ModelSpec]:
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    return {
        name: ModelSpec(
            name=name,
            url=entry["url"],
            sha256=entry["sha256"],
            onnx_sha256=entry["onnx_sha256"],
            license=entry["license"],
            training_data=entry["training_data"],
            member=entry.get("member"),
        )
        for name, entry in data.get("models", {}).items()
    }


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve(spec: ModelSpec, models_dir: Path) -> Path:
    """Path of a verified model file; raises `ModelError` naming the file otherwise."""
    path = models_dir / spec.filename
    if not path.is_file():
        raise ModelError(f"model file {path} not found; run `fim models fetch --only {spec.name}`")
    if sha256_of(path) != spec.onnx_sha256:
        raise ModelError(f"model file {path} does not match its manifest sha256")
    return path


def _urlretrieve(url: str, dest: Path) -> None:
    with urllib.request.urlopen(url, timeout=120) as response, dest.open("wb") as out:
        shutil.copyfileobj(response, out)


def fetch(spec: ModelSpec, models_dir: Path, *, downloader: Downloader = _urlretrieve) -> Path:
    """Download, verify and install one model; nothing is left behind on failure."""
    models_dir.mkdir(parents=True, exist_ok=True)
    dest = models_dir / spec.filename
    with tempfile.TemporaryDirectory(dir=models_dir) as tmp:
        download = Path(tmp) / "download"
        downloader(spec.url, download)
        if sha256_of(download) != spec.sha256:
            raise ModelError(f"download of {spec.name} does not match its manifest sha256")
        candidate = Path(tmp) / spec.filename
        if spec.member:
            with zipfile.ZipFile(download) as archive, archive.open(spec.member) as member:
                candidate.write_bytes(member.read())
        else:
            download.replace(candidate)
        if sha256_of(candidate) != spec.onnx_sha256:
            raise ModelError(f"model file of {spec.name} does not match its manifest sha256")
        candidate.replace(dest)
    return dest
