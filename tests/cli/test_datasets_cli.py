import hashlib
from pathlib import Path

import pytest

from fall_impact_monitor.cli.main import main


def test_list_shows_every_dataset_with_its_terms(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["datasets", "list"]) == 0
    out = capsys.readouterr().out
    for name in ("gmdcsa24", "urfd", "caucafall"):
        assert name in out
    assert "private-use" in out and "CC BY-NC-SA 4.0" in out


def test_fetch_of_a_manual_dataset_explains_what_to_do(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["datasets", "fetch", "caucafall", "--dest", str(tmp_path / "c")]) == 1
    assert "docs/DATASETS.md" in capsys.readouterr().out


def test_fetch_downloads_and_verifies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from fall_impact_monitor.evaluation import datasets

    manifest = tmp_path / "datasets.toml"
    data = b"frames"
    digest = hashlib.sha256(data).hexdigest()
    manifest.write_text(
        "[datasets.demo]\n"
        'title = "Demo"\nversion = "1"\nlicense = "CC BY 4.0"\n'
        'license_url = "https://creativecommons.org/licenses/by/4.0/"\n'
        'attribution = "A. Author"\nstatus = "redistributable"\n'
        'source_url = "https://example.org"\nkind = "files"\n'
        f'files = [{{ path = "v.mp4", url = "https://example.org/v", sha256 = "{digest}" }}]\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(datasets, "_urlretrieve", lambda url, dest: dest.write_bytes(data))
    dest = tmp_path / "out"
    assert (
        main(["datasets", "fetch", "demo", "--dest", str(dest), "--manifest", str(manifest)]) == 0
    )
    assert (dest / "v.mp4").read_bytes() == data
    assert "verified" in capsys.readouterr().out


def test_an_unknown_dataset_is_an_error(tmp_path: Path) -> None:
    assert main(["datasets", "fetch", "nope", "--dest", str(tmp_path)]) == 2
