"""Check Markdown formatting and repository-local links without network access."""

from __future__ import annotations

import os
import subprocess
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import unquote, urlsplit

from markdown_it import MarkdownIt

from tooling.presentations import DECK, GENERATED_HTML
from tooling.requirements import RequirementsError, check_requirements

if TYPE_CHECKING:
    from collections.abc import Iterable

    from markdown_it.token import Token


class DocumentationError(RuntimeError):
    """A repository document contains an invalid local link."""


@dataclass(frozen=True)
class MarkdownDocument:
    anchors: frozenset[str]
    links: tuple[tuple[int, str], ...]


def markdown_paths(root: Path) -> list[Path]:
    """Use the same root and docs tree as the repository's Markdown formatter."""
    return sorted([*root.glob("*.md"), *root.glob("docs/**/*.md")])


def _heading_text(tokens: Iterable[Token]) -> str:
    return "".join(
        token.content
        if token.type in {"text", "code_inline"}
        else _heading_text(token.children or [])
        if token.type == "image"
        else " "
        if token.type in {"softbreak", "hardbreak"}
        else ""
        for token in tokens
    )


def heading_slug(text: str) -> str:
    # GitHub-style Markdown headings retain letters, numbers, marks, '_' and '-'.
    # Punctuation is discarded; spaces become hyphens without being collapsed.
    return "".join(
        "-" if character.isspace() else character
        for character in text.lower()
        if character in "_-"
        or character.isspace()
        or unicodedata.category(character)[0] in "LNM"
    )


def _heading_locations(tokens: list[Token]) -> dict[str, int]:
    anchors: dict[str, int] = {}
    for index, token in enumerate(tokens):
        if (
            token.type != "inline"
            or not index
            or tokens[index - 1].type != "heading_open"
        ):
            continue
        base = heading_slug(_heading_text(token.children or []))
        anchor = base
        suffix = 0
        while anchor in anchors:
            suffix += 1
            anchor = f"{base}-{suffix}"
        anchors[anchor] = token.map[0] + 1 if token.map else 1
    return anchors


def markdown_heading_locations(text: str) -> dict[str, int]:
    """Match link validation, including headings inside quotes and lists."""
    return _heading_locations(MarkdownIt("commonmark").parse(text))


def _read_document(path: Path) -> MarkdownDocument:
    tokens = MarkdownIt("commonmark").parse(path.read_text(encoding="utf-8"))
    anchors = _heading_locations(tokens)
    links: list[tuple[int, str]] = []
    for token in tokens:
        if token.type != "inline":
            continue
        line = token.map[0] + 1 if token.map else 1
        for child in token.children or []:
            attribute = {"link_open": "href", "image": "src"}.get(child.type)
            if attribute is not None:
                destination = child.attrGet(attribute)
                if isinstance(destination, str):
                    links.append((line, destination))
    return MarkdownDocument(frozenset(anchors), tuple(links))


def check_local_links(root: Path, paths: Iterable[Path]) -> None:
    """Validate Markdown links/images and Markdown-heading fragments in this tree.

    External URLs are not fetched. Raw HTML links and anchors are outside this
    check; code examples are ignored by the Markdown parser.
    """
    root = root.resolve()
    documents: dict[Path, MarkdownDocument] = {}
    failures: list[str] = []

    def document(path: Path) -> MarkdownDocument:
        if path not in documents:
            documents[path] = _read_document(path)
        return documents[path]

    for source in paths:
        source = source.resolve()
        if not source.is_relative_to(root):
            raise DocumentationError("Markdown source resolves outside the repository.")
        for line, destination in document(source).links:
            label = f"{source.relative_to(root)}:{line}: {destination}"
            parsed = urlsplit(destination)
            if parsed.scheme or parsed.netloc:
                continue
            path = unquote(parsed.path)
            target = (
                root / path.lstrip("/")
                if path.startswith("/")
                else source.parent / path
                if path
                else source
            ).resolve()
            if not target.is_relative_to(root):
                failures.append(f"{label}: target is outside the repository")
            elif (
                source == root / DECK / "slides.md" and target == root / GENERATED_HTML
            ):
                # The presentation check validates these links against freshly
                # compiled output; a local preview is neither required nor evidence.
                continue
            elif not target.exists():
                failures.append(f"{label}: target does not exist")
            elif (
                parsed.fragment
                and target.is_file()
                and target.suffix.lower() == ".md"
                and unquote(parsed.fragment) not in document(target).anchors
            ):
                failures.append(f"{label}: Markdown heading does not exist")
    if failures:
        raise DocumentationError("\n".join(failures))


def check_documents(root: Path) -> None:
    """Check all repository Markdown, including incoming links to changed files."""
    root = root.resolve()
    paths = markdown_paths(root)
    check_local_links(root, paths)
    requirements = check_requirements(root)
    if paths:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pymarkdown",
                "-d",
                "MD013",
                "scan",
                *(str(path.relative_to(root)) for path in paths),
            ],
            cwd=root,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            check=True,
        )
    print(f"Markdown formatting and local links passed ({len(paths)} documents).")
    if requirements:
        print(f"SRS structure and references passed ({requirements} requirements).")


def main() -> int:
    try:
        check_documents(Path(__file__).resolve().parents[1])
    except subprocess.CalledProcessError as error:
        return error.returncode
    except (DocumentationError, RequirementsError, OSError, ValueError) as error:
        print(f"Documentation checks failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
