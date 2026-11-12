import hashlib
import io
import zipfile
from pathlib import Path

import pytest

from fall_impact_monitor.evaluation.datasets import (
    DatasetEntry,
    DatasetError,
    DatasetFile,
    fetch_dataset,
    load_dataset_manifest,
    split_of,
)

ROOT = Path(__file__).resolve().parents[2]


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def serving(files: dict[str, bytes]):  # type: ignore[no-untyped-def]
    def download(url: str, dest: Path) -> None:
        dest.write_bytes(files[url])

    return download


def entry(**kw: object) -> DatasetEntry:
    base: dict[str, object] = {
        "name": "demo",
        "title": "Demo",
        "version": "1",
        "license": "CC BY 4.0",
        "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "attribution": "Someone et al.",
        "status": "redistributable",
        "source_url": "https://example.org/demo",
        "kind": "files",
        "files": (),
        "instructions": "",
        "splits": {},
    }
    base.update(kw)
    return DatasetEntry(**base)  # type: ignore[arg-type]


def test_the_manifest_describes_the_evaluated_datasets() -> None:
    manifest = load_dataset_manifest(ROOT / "datasets.toml")
    assert {"gmdcsa24", "urfd", "caucafall"} <= set(manifest)
    gmdcsa = manifest["gmdcsa24"]
    assert gmdcsa.kind == "archive" and len(gmdcsa.files) == 1 and len(gmdcsa.files[0].sha256) == 64
    assert len(manifest["urfd"].files) == 72
    assert manifest["urfd"].status == "private-use"  # CC BY-NC-SA
    assert manifest["caucafall"].kind == "manual" and manifest["caucafall"].instructions
    for item in manifest.values():
        assert item.license and item.license_url.startswith("http") and item.attribution
        assert item.status in ("private-use", "redistributable")


def test_files_are_written_only_when_their_sha256_matches(tmp_path: Path) -> None:
    good, bad = b"video", b"tampered"
    demo = entry(
        files=(
            DatasetFile("raw/a.mp4", "https://example.org/a", sha(good)),
            DatasetFile("raw/b.mp4", "https://example.org/b", sha(good)),
        )
    )
    with pytest.raises(DatasetError, match=r"raw/b\.mp4"):
        fetch_dataset(
            demo,
            tmp_path / "demo",
            repo_root=ROOT,
            downloader=serving({"https://example.org/a": good, "https://example.org/b": bad}),
        )
    assert (tmp_path / "demo/raw/a.mp4").read_bytes() == good
    assert not (tmp_path / "demo/raw/b.mp4").exists()
    assert not list((tmp_path / "demo").rglob("*.part"))


def zipped(names: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, data in names.items():
            archive.writestr(name, data)
    return buffer.getvalue()


def test_an_archive_is_verified_and_extracted(tmp_path: Path) -> None:
    data = zipped({"set/Subject 1/Fall/01.mp4": b"v", "set/README.md": b"r"})
    demo = entry(
        kind="archive", files=(DatasetFile("raw/set.zip", "https://example.org/z", sha(data)),)
    )
    fetch_dataset(
        demo, tmp_path / "d", repo_root=ROOT, downloader=serving({"https://example.org/z": data})
    )
    assert (tmp_path / "d/extracted/set/Subject 1/Fall/01.mp4").read_bytes() == b"v"
    assert (tmp_path / "d/raw/set.zip").exists()


@pytest.mark.parametrize("evil", ["../evil.txt", "/abs/evil.txt", "set/../../evil.txt"])
def test_an_archive_escaping_its_folder_is_refused_before_extracting(
    tmp_path: Path, evil: str
) -> None:
    data = zipped({"set/ok.mp4": b"v", evil: b"x"})
    demo = entry(
        kind="archive", files=(DatasetFile("raw/set.zip", "https://example.org/z", sha(data)),)
    )
    with pytest.raises(DatasetError, match="unsafe"):
        fetch_dataset(
            demo,
            tmp_path / "d",
            repo_root=ROOT,
            downloader=serving({"https://example.org/z": data}),
        )
    assert not (tmp_path / "d/extracted").exists()
    assert not (tmp_path / "evil.txt").exists()


def test_a_destination_inside_the_repository_is_refused(tmp_path: Path) -> None:
    with pytest.raises(DatasetError, match="outside the repository"):
        fetch_dataset(entry(), ROOT / "src" / "data", repo_root=ROOT, downloader=serving({}))


def test_the_git_ignored_media_folder_is_allowed(tmp_path: Path) -> None:
    demo = entry(files=(DatasetFile("a.bin", "https://example.org/a", sha(b"x")),))
    fake_root = tmp_path / "repo"
    (fake_root / "test_videos").mkdir(parents=True)
    fetch_dataset(
        demo,
        fake_root / "test_videos" / "datasets" / "demo",
        repo_root=fake_root,
        downloader=serving({"https://example.org/a": b"x"}),
    )
    assert (fake_root / "test_videos/datasets/demo/a.bin").exists()


def test_a_manual_dataset_is_not_fetched(tmp_path: Path) -> None:
    with pytest.raises(DatasetError, match="by hand"):
        fetch_dataset(entry(kind="manual", instructions="see docs"), tmp_path / "x", repo_root=ROOT)


def test_splits_are_by_subject_and_disjoint() -> None:
    manifest = load_dataset_manifest(ROOT / "datasets.toml")
    gmdcsa = manifest["gmdcsa24"]
    assert split_of(gmdcsa, "Subject 1/Fall/01.mp4") == "train"
    assert split_of(gmdcsa, "Subject 4/ADL/02.mp4") == "test"
    for item in manifest.values():
        seen: set[str] = set()
        for prefixes in item.splits.values():
            assert seen.isdisjoint(prefixes)
            seen |= set(prefixes)
