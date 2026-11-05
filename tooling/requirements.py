"""Validate the maintained SRS structure without executing its evidence links."""

from __future__ import annotations

import posixpath
import re
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlsplit

from markdown_it import MarkdownIt

INDEX = "docs/requirements.md"
DIRECTORY = "docs/requirements"
# One registry drives both validation and immutable review-context selection.
# The first scope owns changes; the tuple identifies all affected consumers.
DOCUMENT_SCOPES = {
    "docs/requirements/strategy.md": ("strategy", "execution", "interfaces"),
    "docs/requirements/execution.md": (
        "execution",
        "strategy",
        "results",
        "interfaces",
    ),
    "docs/requirements/data.md": ("data", "execution", "ml", "interfaces"),
    "docs/requirements/results.md": ("results", "execution", "ml", "interfaces"),
    "docs/requirements/ml.md": ("ml", "strategy", "results", "interfaces"),
    "docs/requirements/interfaces.md": ("interfaces",),
    "docs/requirements/quality.md": (
        "interfaces",
        "strategy",
        "execution",
        "results",
        "data",
        "ml",
    ),
}
_ID = re.compile(r"(?:FR|NFR)-[A-Z][A-Z0-9]*-[0-9]{3}")
_FIELD = re.compile(r"\*\*(需求|驗收|契約|驗證)\*\*：[ \t]*(.*)", re.DOTALL)
_FIELDS = frozenset({"需求", "驗收", "契約", "驗證"})


class RequirementsError(RuntimeError):
    """SRS identifiers, acceptance records or trace references are invalid."""


def is_requirements_path(name: str) -> bool:
    """Include additions, deletions and both sides of renames in candidate review."""
    path = PurePosixPath(name)
    return name == INDEX or (path.is_relative_to(DIRECTORY) and path.suffix == ".md")


def contract_references(source: str, text: str) -> set[tuple[str, str]]:
    """Resolve SRS contract links lexically for immutable Git evidence readers."""
    references: set[tuple[str, str]] = set()
    for token in MarkdownIt("commonmark").parse(text):
        if token.type != "inline" or token.level != 3:
            continue
        match = _FIELD.fullmatch(token.content)
        if match is None or match[1] != "契約":
            continue
        for child in token.children or []:
            href = child.attrGet("href") if child.type == "link_open" else None
            if not isinstance(href, str):
                continue
            parsed = urlsplit(href)
            if parsed.scheme or parsed.netloc:
                continue
            path = unquote(parsed.path)
            resolved = posixpath.normpath(
                path.lstrip("/")
                if path.startswith("/")
                else posixpath.join(posixpath.dirname(source), path)
                if path
                else source
            )
            if resolved.startswith("../") or not resolved.endswith(".md"):
                raise RequirementsError(f"{source}: invalid contract reference: {href}")
            if is_requirements_path(resolved):
                raise RequirementsError(
                    f"{source}: SRS cannot replace its contract: {href}"
                )
            references.add((resolved, unquote(parsed.fragment)))
    return references


def _target(root: Path, source: Path, href: str) -> Path | None:
    parsed = urlsplit(href)
    if parsed.scheme or parsed.netloc:
        return None
    path = unquote(parsed.path)
    target = (
        root / path.lstrip("/")
        if path.startswith("/")
        else source.parent / path
        if path
        else source
    ).resolve()
    if not target.is_relative_to(root) or not target.is_file():
        raise RequirementsError(f"{source}: invalid local evidence reference: {href}")
    return target


def check_requirements(root: Path) -> int:
    """Check records and navigation; links are evidence targets, not test results.

    Ordinary Markdown link/anchor checks remain owned by tooling.documentation.
    An absent SRS is allowed for repositories/fixtures predating its introduction;
    a present entry or chapter requires the complete connected document set.
    """
    root = root.resolve()
    index = root / INDEX
    chapters = sorted((root / DIRECTORY).rglob("*.md"))
    if not index.exists() and not chapters:
        return 0
    if not index.is_file() or not chapters:
        raise RequirementsError(
            "SRS requires an entry document and requirement chapters"
        )
    for source in [index, *chapters]:
        if not source.resolve().is_relative_to(root):
            raise RequirementsError(f"SRS source resolves outside repository: {source}")
    for source in chapters:
        if source.relative_to(root).as_posix() not in DOCUMENT_SCOPES:
            raise RequirementsError(f"{source}: unregistered SRS review responsibility")
    parser = MarkdownIt("commonmark")
    navigation = {
        _target(root, index, href)
        for token in parser.parse(index.read_text(encoding="utf-8"))
        for child in token.children or []
        if child.type == "link_open" and isinstance(href := child.attrGet("href"), str)
    }
    missing = [
        str(path.relative_to(root)) for path in chapters if path not in navigation
    ]
    if missing:
        raise RequirementsError(
            f"SRS entry does not link chapters: {', '.join(missing)}"
        )
    identifiers: dict[str, str] = {}
    for source in chapters:
        _check_chapter(root, source, identifiers)
    return len(identifiers)


def _finish(source: Path, current: str | None, fields: set[str]) -> None:
    if current is not None and fields != _FIELDS:
        raise RequirementsError(
            f"{source}: {current}: missing fields {', '.join(sorted(_FIELDS - fields))}"
        )


def _check_chapter(root: Path, source: Path, identifiers: dict[str, str]) -> None:
    parser = MarkdownIt("commonmark")
    text = source.read_text(encoding="utf-8")
    contract_references(source.relative_to(root).as_posix(), text)
    tokens = parser.parse(text)
    current: str | None = None
    fields: set[str] = set()
    count = 0

    for position, token in enumerate(tokens):
        if token.type != "inline":
            continue
        if position and tokens[position - 1].type == "heading_open":
            heading = tokens[position - 1]
            if heading.level != 0:
                continue
            _finish(source, current, fields)
            current = None
            fields = set()
            title = token.content
            if not title.startswith(("FR-", "NFR-")):
                continue
            identifier, separator, label = title.partition(" ")
            if (
                heading.tag != "h2"
                or not _ID.fullmatch(identifier)
                or not separator
                or not label.strip()
            ):
                raise RequirementsError(
                    f"{source}: malformed requirement heading: {title}"
                )
            location = f"{source.relative_to(root)}:{(token.map or [0])[0] + 1}"
            if identifier in identifiers:
                raise RequirementsError(
                    f"{location}: duplicate {identifier}; "
                    f"first at {identifiers[identifier]}"
                )
            identifiers[identifier] = location
            current = identifier
            count += 1
            continue
        if current is None or token.level != 3:
            continue
        match = _FIELD.fullmatch(token.content)
        if match is None:
            continue
        name, content = match.groups()
        if name in fields or not content.strip():
            raise RequirementsError(f"{source}: {current}: empty or duplicate {name}")
        fields.add(name)
        if name not in {"契約", "驗證"}:
            continue
        targets = [
            target
            for child in token.children or []
            if child.type == "link_open"
            and isinstance(href := child.attrGet("href"), str)
            and (target := _target(root, source, href)) is not None
        ]
        if name == "契約":
            valid = any(
                path.suffix == ".md"
                and not is_requirements_path(path.relative_to(root).as_posix())
                for path in targets
            )
        else:
            valid = any(
                path.is_relative_to(root / "tests")
                and path.name.startswith("test_")
                and path.suffix == ".py"
                for path in targets
            ) or (content.startswith("人工驗收：") and bool(content[5:].strip()))
        if not valid:
            raise RequirementsError(
                f"{source}: {current}: missing {name} evidence target"
            )
    _finish(source, current, fields)
    if not count:
        raise RequirementsError(f"{source}: no requirement records")
