"""A portable deck keeps images, report links and download artifacts together."""

from __future__ import annotations

import shutil
import subprocess
import zipfile
from io import BytesIO
from pathlib import Path

import pytest
from tests.tooling.test_presentations import _html
from tooling.presentation_bundle import (
    check_bundle,
    evidence_content_allowed,
    write_bundle,
)
from tooling.presentation_errors import PresentationError
from tooling.presentation_html import check_html
from tooling.presentations import DECK, presentation_change_allowed

PNG = b"\x89PNG\r\n\x1a\nfixture"


def _delivery(root: Path) -> Path:
    root.mkdir(parents=True)
    (root / "assets").mkdir()
    (root / "assets/chart.png").write_bytes(PNG)
    (root / "evidence").mkdir()
    (root / "evidence/report.html").write_text(
        '<h1 id="result">Saved result</h1><a href="../index.html#2">Back</a>'
    )
    (root / "evidence/research.zip").write_bytes(b"saved download")
    (root / "index.html").write_text(
        _html(
            '<img src="assets/chart.png">'
            '<a href="evidence/report.html#result">Report</a>'
        )
    )
    return root


def test_folder_and_zip_keep_the_complete_readable_delivery(tmp_path: Path) -> None:
    source = _delivery(tmp_path / "source")
    (source / "slides.md").write_text("Authoring source is not a delivery entry")
    (source / "private").mkdir()
    (source / "private/secret.txt").write_text("not published")
    output = tmp_path / "delivery"
    archive = write_bundle(source, output)
    assert check_bundle(output) == 4
    assert check_html((output / "index.html").read_text(), root=output) == 2
    assert not (output / "slides.md").exists()
    assert not (output / "private").exists()
    with zipfile.ZipFile(archive) as package:
        assert package.testzip() is None
        package.extractall(tmp_path / "extracted")
        assert package.read("delivery/assets/chart.png") == PNG
        assert (
            package.read("delivery/evidence/report.html")
            == (source / "evidence/report.html").read_bytes()
        )
    assert check_bundle(tmp_path / "extracted/delivery") == 4


@pytest.mark.parametrize(
    "url",
    [
        "assets/missing.png",
        "../outside.png",
        "/outside.png",
        "https://example.com/a.png",
    ],
)
def test_missing_or_nonlocal_deck_images_fail(tmp_path: Path, url: str) -> None:
    source = _delivery(tmp_path / "source")
    (tmp_path / "outside.png").write_bytes(PNG)
    with pytest.raises(PresentationError):
        check_html(_html(f'<img src="{url}">'), root=source)


@pytest.mark.parametrize(
    "url",
    [
        "evidence/missing.html",
        "evidence/report.html#missing",
        "slides.md",
        "../outside.html",
    ],
)
def test_missing_unpublished_or_escaping_report_links_fail(
    tmp_path: Path, url: str
) -> None:
    source = _delivery(tmp_path / "source")
    (source / "slides.md").write_text("not packaged")
    (tmp_path / "outside.html").write_text("not packaged")
    (source / "index.html").write_text(_html(f'<a href="{url}">Report</a>'))
    with pytest.raises(PresentationError):
        write_bundle(source, tmp_path / "delivery")
    assert not (tmp_path / "delivery").exists()


def test_linked_files_are_never_copied_or_published(tmp_path: Path) -> None:
    source = _delivery(tmp_path / "source")
    (source / "assets/chart.png").unlink()
    (tmp_path / "outside.png").write_bytes(PNG)
    (source / "assets/chart.png").symlink_to(tmp_path / "outside.png")
    with pytest.raises(PresentationError, match="symlinks"):
        write_bundle(source, tmp_path / "delivery")


def test_failed_copy_cleans_its_output_and_preserves_existing_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _delivery(tmp_path / "source")
    output = tmp_path / "delivery"

    def fail_copy(*args: object, **kwargs: object) -> None:
        raise OSError("copy failed")

    monkeypatch.setattr(shutil, "copyfile", fail_copy)
    with pytest.raises(OSError, match="copy failed"):
        write_bundle(source, output)
    assert not output.exists() and not output.with_suffix(".zip").exists()
    output.with_suffix(".zip").write_bytes(b"keep me")
    with pytest.raises(PresentationError, match="already exist"):
        write_bundle(source, output)
    assert output.with_suffix(".zip").read_bytes() == b"keep me"


def test_asset_classification_rejects_code_disguised_as_an_image() -> None:
    assert presentation_change_allowed(f"{DECK}/assets/chart.png", None, PNG)
    assert not presentation_change_allowed(f"{DECK}/assets/chart.png", None, b"code")
    assert not presentation_change_allowed(f"{DECK}/assets/run.py", None, PNG)


@pytest.mark.parametrize(
    "name, content",
    [
        ("demo/index.html", b"<h1>Case</h1>"),
        ("demo/case.json", b'{"case": {}}'),
        ("demo/checksums.json", b"{}"),
        ("demo/cases/example/benchmark.json", b'{"status": "unavailable"}'),
        ("demo/cases/example/confirmation.txt", "確認單".encode()),
        ("demo/cases/example/report.html", b"<script>/* native report */</script>"),
        ("demo/cases/example/strategy.py", b"raise RuntimeError('never execute')"),
        ("demo/cases/example/notebook/research.json", b"{}"),
        ("demo/cases/example/notebook/reports/12.html", b"<h1>Report</h1>"),
        (f"demo/cases/example/earlier-reports/{'a' * 64}.html", b"<h1>Report</h1>"),
        ("demo/screenshots/01-native.png", PNG),
        (
            "demo/.gitattributes",
            b"# Preserve native bytes\n"
            b"cases/example/report.html whitespace=-blank-at-eol\n",
        ),
    ],
)
def test_saved_document_roles_are_read_only(name: str, content: bytes) -> None:
    assert evidence_content_allowed(name, content)


def test_archive_role_checks_format_without_extracting_or_executing() -> None:
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("strategy.py", "raise RuntimeError('never execute')")
    for name in ("demo/natural-case-evidence.zip", "demo/cases/example/research.zip"):
        assert evidence_content_allowed(name, buffer.getvalue())
        assert not evidence_content_allowed(name, b"not a ZIP")


@pytest.mark.parametrize(
    "name, content",
    [
        ("demo/cases/example/run.py", b"print('unknown program')"),
        ("demo/cases/example/plugin.js", b"alert('unknown program')"),
        ("demo/cases/example/unknown.json", b"{}"),
        ("demo/cases/example/benchmark.json", b"{invalid"),
        ("demo/cases/example/report.html", b"\xff"),
        ("demo/cases/example/strategy.py", b"binary\0payload"),
        ("demo/cases/example/notebook/reports/latest.html", b"<h1>Unknown role</h1>"),
        ("demo/cases/example/earlier-reports/latest.html", b"<h1>Unknown role</h1>"),
        ("demo/screenshots/01-native.png", b"disguised program"),
        ("demo/screenshots/nested/01-native.png", PNG),
        ("../demo/index.html", b"<h1>Outside</h1>"),
        ("demo/cases/example/../report.html", b"<h1>Outside</h1>"),
        ("demo/.gitattributes", b"*.html whitespace=-blank-at-eol"),
        ("demo/.gitattributes", b"cases/example/report.html filter=custom"),
        ("demo/.gitattributes", b"cases/example/strategy.py whitespace=-blank-at-eol"),
    ],
)
def test_unknown_or_malformed_evidence_retains_code_checks(
    name: str, content: bytes
) -> None:
    assert not evidence_content_allowed(name, content)


def test_cloudflare_stages_all_entries_and_removes_only_stale_output(
    tmp_path: Path,
) -> None:
    script = Path(__file__).resolve().parents[2] / "scripts/stage-presentations.sh"
    (tmp_path / "scripts").mkdir()
    shutil.copyfile(script, tmp_path / "scripts/stage-presentations.sh")
    source = _delivery(tmp_path / DECK)
    target = tmp_path / "site-dist/tradingdev"
    target.mkdir(parents=True)
    (target / "stale.txt").write_text("old publication")
    sibling = tmp_path / "site-dist/other-site.txt"
    sibling.write_text("keep me")
    subprocess.run(
        ["bash", str(tmp_path / "scripts/stage-presentations.sh")],
        check=True,
        timeout=10,
    )
    assert check_bundle(target) == 4
    assert not (target / "stale.txt").exists()
    assert sibling.read_text() == "keep me"
    assert (target / "assets/chart.png").read_bytes() == PNG
    assert not list((tmp_path / "site-dist").glob(".tradingdev.*"))
    shutil.rmtree(source / "assets")
    failed = subprocess.run(
        ["bash", str(tmp_path / "scripts/stage-presentations.sh")],
        check=False,
        timeout=10,
    )
    assert failed.returncode != 0
    assert check_bundle(target) == 4
