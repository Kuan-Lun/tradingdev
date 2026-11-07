"""Verify offline resources and page identity in generated presentation HTML."""

from __future__ import annotations

import re
from html.parser import HTMLParser
from typing import TYPE_CHECKING

from tooling.presentation_bundle import check_image
from tooling.presentation_errors import PresentationError

if TYPE_CHECKING:
    from pathlib import Path


def _css_unescape(value: str) -> str:
    def replace(match: re.Match[str]) -> str:
        escaped = match[1]
        if re.fullmatch(r"[0-9a-fA-F]{1,6}\s?", escaped):
            number = int(escaped.strip(), 16)
            return chr(number) if 0 < number <= 0x10FFFF else "\ufffd"
        return escaped

    return re.sub(r"\\([0-9a-fA-F]{1,6}\s?|[^\n\r\f])", replace, value)


def _check_css(value: str, root: Path | None = None) -> None:
    css = _css_unescape(re.sub(r"/\*.*?\*/", "", value, flags=re.DOTALL))
    if re.search(r"@import\b", css, flags=re.IGNORECASE):
        raise PresentationError("External CSS imports are not allowed.")
    if re.search(r"\b(?:image-set|image)\s*\(", css, flags=re.IGNORECASE):
        raise PresentationError(
            "CSS image functions with implicit URLs are not allowed."
        )
    for match in re.finditer(r"\burl\s*\(\s*([^)]*)\)", css, flags=re.IGNORECASE):
        target = match[1].strip().strip("\"'").strip()
        if target and not target.casefold().startswith(("data:", "#")):
            if root is None:
                raise PresentationError(
                    f"External CSS resource is not allowed: {target}"
                )
            check_image(root, target)


class _PresentationHTML(HTMLParser):
    def __init__(self, root: Path | None = None) -> None:
        super().__init__(convert_charrefs=True)
        self.root = root
        self.sections: list[dict[str, str | None]] = []
        self.active: str | None = None
        self.content: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if tag in {"iframe", "object", "embed"}:
            raise PresentationError(f"Embedded active content is not allowed: {tag}")
        if tag == "section":
            self.sections.append(attributes)
        if tag in {"style", "script"}:
            self.active, self.content = tag, []
        if tag == "base" or (
            tag == "meta" and (attributes.get("http-equiv") or "").lower() == "refresh"
        ):
            raise PresentationError(f"Unexpected navigation element: {tag}")
        for name, value in attrs:
            if name == "style" and value:
                _check_css(value, self.root)
            if name in {"srcdoc", "srcset", "ping"} or name.startswith("on"):
                raise PresentationError(f"Inline executable HTML attribute: {name}")
            resource = name in {"src", "srcset", "poster", "data", "background"} or (
                name in {"href", "xlink:href"} and tag not in {"a"}
            )
            if (
                resource
                and value
                and not value.strip().lower().startswith(("data:", "#"))
            ):
                if self.root is None or not (
                    (tag == "img" and name == "src")
                    or (tag == "image" and name in {"href", "xlink:href"})
                ):
                    raise PresentationError(
                        f"External HTML resource is not allowed: {value}"
                    )
                check_image(self.root, value)
            if (
                tag == "a"
                and name == "href"
                and value
                and re.match(r"\s*(?:javascript|data):", value, flags=re.IGNORECASE)
            ):
                raise PresentationError("Executable citation URLs are not allowed.")

    def handle_data(self, data: str) -> None:
        if self.active:
            self.content.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag != self.active:
            return
        content = "".join(self.content)
        if tag == "style":
            _check_css(content, self.root)
        elif re.search(
            r"WebSocket|EventSource|location\s*\.\s*reload|marp-cli-reload|wss?://",
            content,
            flags=re.IGNORECASE,
        ):
            raise PresentationError(
                "Watch/reload code is not allowed in committed HTML."
            )
        self.active, self.content = None, []


def check_html(content: str, *, root: Path | None = None) -> int:
    """Inspect generated output as data; never open it in a browser."""
    parser = _PresentationHTML(root)
    parser.feed(content)
    parser.close()
    count = len(parser.sections)
    if not count:
        raise PresentationError("The presentation contains no sections.")
    for number, section in enumerate(parser.sections, 1):
        if (
            section.get("id") != str(number)
            or section.get("data-marpit-pagination") != str(number)
            or section.get("data-marpit-pagination-total") != str(count)
        ):
            raise PresentationError(
                "Section IDs, page numbers and totals are inconsistent."
            )
    return count
