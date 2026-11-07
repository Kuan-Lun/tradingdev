"""Validate and package the published deck and its local evidence as one unit."""

from __future__ import annotations

import shutil
import zipfile
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

from tooling.presentation_errors import PresentationError

BUNDLE_ENTRIES = ("index.html", "assets", "evidence")
IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".gif", ".webp"})


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
