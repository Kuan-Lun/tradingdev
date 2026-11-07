"""Validate and package the published deck and its local evidence as one unit."""

from __future__ import annotations

import json
import re
import shutil
import zipfile
from html.parser import HTMLParser
from io import BytesIO
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlsplit

from tooling.presentation_errors import PresentationError

BUNDLE_ENTRIES = ("index.html", "assets", "evidence")
IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".gif", ".webp"})

# Saved demonstration documents, never imported by the product or executed by
# presentation checks. New artifact roles require registration here.
_CASE_DOCUMENTS = frozenset(
    {
        "analysis.json",
        "benchmark.json",
        "commentary.json",
        "confirmation.html",
        "confirmation.txt",
        "critique.html",
        "dialogue.html",
        "execution-manifest.json",
        "metrics.json",
        "provenance.json",
        "report.html",
        "research.zip",
        "strategy.py",
        "trades.json",
        "transcript.json",
        "conversation/index.html",
        "conversation/conversation.json",
        "conversation/manifest.json",
        "notebook/index.html",
        "notebook/manifest.json",
        "notebook/research.json",
    }
)
_IDENTIFIER = r"[A-Za-z0-9][A-Za-z0-9_-]*"


def _case_document(path: str) -> bool:
    parts = path.split("/", 2)
    return (
        len(parts) == 3
        and parts[0] == "cases"
        and re.fullmatch(_IDENTIFIER, parts[1]) is not None
        and (
            parts[2] in _CASE_DOCUMENTS
            or re.fullmatch(r"earlier-reports/[0-9a-f]{64}\.html", parts[2]) is not None
            or re.fullmatch(r"notebook/reports/[1-9][0-9]*\.html", parts[2]) is not None
        )
    )


def evidence_content_allowed(path: str, content: bytes) -> bool:
    """Recognize registered saved-document roles, not arbitrary evidence files.

    The caller checks both Git versions and ordinary file modes. Format checks
    establish the role only; links, semantic accuracy and browser behavior still
    need their respective presentation checks and review.
    """
    parts = path.split("/", 1)
    if len(parts) != 2 or re.fullmatch(_IDENTIFIER, parts[0]) is None:
        return False
    name = parts[1]
    if name.startswith("screenshots/"):
        filename = name.removeprefix("screenshots/")
        suffix = PurePosixPath(filename).suffix
        return re.fullmatch(
            _IDENTIFIER, filename.removesuffix(suffix)
        ) is not None and is_image(content, suffix)
    if name not in {
        ".gitattributes",
        "case.json",
        "checksums.json",
        "index.html",
        "natural-case-evidence.zip",
    } and not _case_document(name):
        return False
    if name.endswith(".zip"):
        return zipfile.is_zipfile(BytesIO(content))
    try:
        text = content.decode("utf-8")
        if "\0" in text:
            return False
        if name.endswith(".json"):
            json.loads(text)
        if name == ".gitattributes":
            for line in text.splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                fields = line.split()
                if (
                    len(fields) != 2
                    or fields[1] != "whitespace=-blank-at-eol"
                    or not _case_document(fields[0])
                    or not fields[0].endswith(".html")
                ):
                    return False
    except ValueError, UnicodeError:
        return False
    return True


def is_image(content: bytes, suffix: str) -> bool:
    """Recognize the raster formats permitted in the deck's asset directory."""
    if suffix in {".jpg", ".jpeg"}:
        return content.startswith(b"\xff\xd8\xff")
    if suffix == ".png":
        return content.startswith(b"\x89PNG\r\n\x1a\n")
    if suffix == ".gif":
        return content.startswith((b"GIF87a", b"GIF89a"))
    return suffix == ".webp" and content[:4] == b"RIFF" and content[8:12] == b"WEBP"


def local_target(root: Path, origin: Path, url: str) -> Path:
    """Resolve a file URL relative to its page, confined to the delivery root."""
    parts = urlsplit(url)
    path = unquote(parts.path)
    if parts.scheme or parts.netloc or path.startswith("/") or "\\" in path:
        raise PresentationError(f"Expected a bundle-relative URL: {url}")
    target = origin.parent / path if path else origin
    if not target.resolve().is_relative_to(root.resolve()) or not target.is_file():
        raise PresentationError(f"Missing or escaping bundle target: {origin}: {url}")
    return target.resolve()


def check_image(root: Path, url: str) -> None:
    target = local_target(root, root / "index.html", url)
    if not target.is_relative_to((root / "assets").resolve()) or not is_image(
        target.read_bytes(), target.suffix.lower()
    ):
        raise PresentationError(f"Expected a raster image under assets/: {url}")


def bundle_files(root: Path) -> list[Path]:
    """Only explicit delivery entries are publishable; never follow symlinks."""
    files: list[Path] = []
    for name in BUNDLE_ENTRIES:
        entry = root / name
        if entry.is_symlink():
            raise PresentationError(f"Bundle symlinks are not allowed: {entry}")
        if not entry.exists() and name == "index.html":
            raise PresentationError("The bundle has no index.html")
        if not entry.exists():
            continue
        for path in [entry, *sorted(entry.rglob("*"))] if entry.is_dir() else [entry]:
            if path.is_symlink():
                raise PresentationError(f"Bundle symlinks are not allowed: {path}")
            if path.is_file():
                files.append(path)
    return files


class _Links(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[tuple[str, bool]] = []
        self.ids: set[str] = set()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if identity := attributes.get("id"):
            self.ids.add(identity)
        for name in ("href", "src", "poster", "xlink:href"):
            if value := attributes.get(name):
                self.links.append((value, tag != "a"))


def check_bundle(root: Path) -> int:
    """Check local HTML links and fragments without executing report scripts."""
    paths = bundle_files(root)
    published = {p.resolve() for p in paths}
    pages: dict[Path, _Links] = {}
    for path in paths:
        if path.suffix.lower() in {".html", ".htm"}:
            parsed = _Links()
            parsed.feed(path.read_text(encoding="utf-8"))
            parsed.close()
            pages[path.resolve()] = parsed
    for page, parsed in pages.items():
        for url, resource in parsed.links:
            parts = urlsplit(url)
            if parts.scheme or parts.netloc:
                if resource and parts.scheme != "data":
                    raise PresentationError(f"Remote bundle resource: {page}: {url}")
                if not resource and parts.scheme not in {"http", "https", "mailto"}:
                    raise PresentationError(f"Unsupported bundle link: {page}: {url}")
                continue
            target = local_target(root, page, url)
            if target not in published:
                raise PresentationError(f"Link target is not packaged: {page}: {url}")
            if (
                parts.fragment
                and target in pages
                and unquote(parts.fragment) not in pages[target].ids
            ):
                raise PresentationError(f"Missing bundle fragment: {page}: {url}")
    return len(paths)


def write_bundle(root: Path, output: Path) -> Path:
    """Create a fresh delivery folder and ZIP; the source research stays intact."""
    check_bundle(root)
    if output.resolve().is_relative_to(root.resolve()):
        raise PresentationError("Bundle output must be outside its source directory")
    archive = output.with_suffix(".zip")
    if archive.exists() or archive.is_symlink():
        raise PresentationError(f"Bundle archive must not already exist: {archive}")
    if output.exists() or output.is_symlink():
        raise PresentationError(f"Bundle output must not already exist: {output}")
    output.mkdir(parents=True)
    archive_created = False
    try:
        for source in bundle_files(root):
            destination = output / source.relative_to(root)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
        check_bundle(output)
        with zipfile.ZipFile(archive, "x", zipfile.ZIP_DEFLATED) as bundle:
            archive_created = True
            for source in bundle_files(output):
                bundle.write(source, Path(output.name) / source.relative_to(output))
    except BaseException:
        shutil.rmtree(output)
        if archive_created:
            archive.unlink()
        raise
    return archive
