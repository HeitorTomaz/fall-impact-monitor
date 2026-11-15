"""Dataset manifest (`datasets.toml`) and user-initiated downloads (`fim datasets`).

Datasets are never part of the repository and are never redistributed. Downloads go to
a folder outside the repository (the git-ignored `test_videos/` is accepted), every
file is checked against the SHA-256 in the manifest before it is kept, and archives are
extracted only when none of their entries would leave the destination folder. Archive
contents are data: nothing in them is ever executed.
"""

import hashlib
import shutil
import tomllib
import urllib.request
import zipfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

Downloader = Callable[[str, Path], None]
_IGNORED_MEDIA = "test_videos"  # git-ignored folder inside the repository
_CHUNK = 1 << 20


class DatasetError(Exception):
    """A dataset could not be fetched safely."""


@dataclass(frozen=True)
class DatasetFile:
    path: str  # relative to the dataset folder
    url: str
    sha256: str


@dataclass(frozen=True)
class DatasetEntry:
    name: str
    title: str
    version: str
    license: str
    license_url: str
    attribution: str
    status: str  # "private-use" or "redistributable"
    source_url: str
    kind: str  # "archive", "files" or "manual"
    files: tuple[DatasetFile, ...] = ()
    instructions: str = ""
    splits: Mapping[str, tuple[str, ...]] = field(default_factory=dict)  # name prefixes


def load_dataset_manifest(path: Path) -> dict[str, DatasetEntry]:
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    manifest: dict[str, DatasetEntry] = {}
    for name, raw in data["datasets"].items():
        base_url = raw.get("base_url", "")
        files = tuple(
            DatasetFile(
                f["path"],
                f.get("url") or (base_url + f["path_on_server"] if "path_on_server" in f else ""),
                f["sha256"],
            )
            for f in raw.get("files", [])
        )
        manifest[name] = DatasetEntry(
            name=name,
            title=raw["title"],
            version=str(raw["version"]),
            license=raw["license"],
            license_url=raw["license_url"],
            attribution=raw["attribution"],
            status=raw["status"],
            source_url=raw["source_url"],
            kind=raw["kind"],
            files=files,
            instructions=raw.get("instructions", ""),
            splits={k: tuple(v) for k, v in raw.get("splits", {}).items()},
        )
    return manifest


def split_of(entry: DatasetEntry, video_name: str) -> str | None:
    """The split (train, validation, test) of a video, by its name's prefix."""
    for split, prefixes in entry.splits.items():
        if any(video_name.startswith(prefix) for prefix in prefixes):
            return split
    return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _urlretrieve(url: str, dest: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "fall-impact-monitor"})
    with urllib.request.urlopen(request) as response, dest.open("wb") as out:
        shutil.copyfileobj(response, out, _CHUNK)


def _check_destination(dest: Path, repo_root: Path) -> Path:
    dest, root = dest.resolve(), repo_root.resolve()
    if dest.is_relative_to(root) and not dest.is_relative_to(root / _IGNORED_MEDIA):
        raise DatasetError(
            f"write datasets outside the repository (or under {_IGNORED_MEDIA}/), not {dest}"
        )
    return dest


def _inside(folder: Path, relative: str) -> Path:
    target = (folder / relative).resolve()
    if not target.is_relative_to(folder.resolve()):
        raise DatasetError(f"unsafe path {relative!r}")
    return target


def _safe_members(archive: zipfile.ZipFile, folder: Path) -> list[tuple[zipfile.ZipInfo, Path]]:
    """Every entry with its target, checked before anything is written."""
    members = []
    for info in archive.infolist():
        name = PurePosixPath(info.filename)
        if name.is_absolute() or ".." in name.parts or info.filename.startswith(("/", "\\")):
            raise DatasetError(f"unsafe archive entry {info.filename!r}")
        if (info.external_attr >> 16) & 0o170000 == 0o120000:  # a symbolic link
            raise DatasetError(f"unsafe archive entry {info.filename!r} (link)")
        members.append((info, _inside(folder, info.filename)))
    return members


def _extract(archive_path: Path, folder: Path) -> None:
    with zipfile.ZipFile(archive_path) as archive:
        members = _safe_members(archive, folder)
        for info, target in members:
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as source, target.open("wb") as out:
                shutil.copyfileobj(source, out, _CHUNK)


def fetch_dataset(
    entry: DatasetEntry,
    dest: Path,
    *,
    repo_root: Path,
    downloader: Downloader = _urlretrieve,
) -> Path:
    """Download (and for archives extract) ``entry`` into ``dest``; files already
    present with the right SHA-256 are kept. Raises `DatasetError` on any mismatch.

    A ``manual`` dataset is downloaded by hand into ``dest``: its listed files are
    verified, and its zip files extracted, never downloaded."""
    if entry.kind == "manual":
        return _verify_by_hand(entry, _check_destination(dest, repo_root))
    dest = _check_destination(dest, repo_root)
    for item in entry.files:
        target = _inside(dest, item.path)
        if not (target.exists() and _sha256(target) == item.sha256):
            target.parent.mkdir(parents=True, exist_ok=True)
            partial = target.with_name(target.name + ".part")
            try:
                downloader(item.url, partial)
                if _sha256(partial) != item.sha256:
                    raise DatasetError(f"{item.path} does not match its manifest sha256")
                partial.replace(target)
            finally:
                partial.unlink(missing_ok=True)
        if entry.kind == "archive":
            _extract(target, dest / "extracted")
    return dest


def _verify_by_hand(entry: DatasetEntry, dest: Path) -> Path:
    steps = f"{entry.name} must be downloaded by hand: {entry.instructions}"
    if not entry.files:
        raise DatasetError(steps)
    for item in entry.files:
        target = _inside(dest, item.path)
        if not target.exists():
            raise DatasetError(f"{item.path} is missing; {steps}")
        if _sha256(target) != item.sha256:
            raise DatasetError(f"{item.path} does not match its manifest sha256")
    for item in entry.files:
        if item.path.endswith(".zip"):
            _extract(_inside(dest, item.path), dest / "extracted")
    return dest


def describe(entry: DatasetEntry) -> dict[str, Any]:
    return {
        "name": entry.name,
        "version": entry.version,
        "license": entry.license,
        "status": entry.status,
        "kind": entry.kind,
        "files": len(entry.files),
        "source": entry.source_url,
    }
