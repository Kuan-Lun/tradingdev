"""Shared document rendering preserves evidence and HTML trust boundaries."""

from __future__ import annotations

import html
import json
from html.parser import HTMLParser

import pytest

from tradingdev.adapters.presentation.html import (
    details_html,
    pairs_html,
    render_html_document,
    value_html,
    value_text,
)


class _DocumentParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.scripts: list[dict[str, str | None]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "script":
            self.scripts.append(dict(attrs))


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, "未知／未記錄"),
        (0, "0"),
        (False, "False"),
        ("", ""),
        ("<>&\"'", "<>&\"'"),
    ],
)
def test_html_values_preserve_plain_text_and_missingness(
    value: object, expected: str
) -> None:
    assert value_text(value) == expected
    assert html.unescape(value_html(value)) == expected


def test_structured_values_and_labels_are_escaped_without_losing_content() -> None:
    value = {"風險": [0, None, "</script><img src=x onerror=alert(1)>"]}
    assert json.loads(value_text(value)) == value
    rendered = pairs_html({"<script>label</script>": value})
    assert "<script>" not in rendered and "<img" not in rendered
    assert "&lt;script&gt;label&lt;/script&gt;" in rendered
    assert "未知／未記錄" not in rendered  # Nested null remains JSON null.
    assert "<img" not in details_html("<img>", value)


def test_document_escapes_title_attributes_and_script_data() -> None:
    attack = "</script><script>alert(1)</script><img src=x>"
    data = {"值": attack + "&\u2028\u2029"}
    content = render_html_document(
        title=attack,
        body="<section><h1>確認設定</h1></section>",
        data=data,
        data_id="x' onerror='alert(1)",
    )
    parser = _DocumentParser()
    parser.feed(content)
    assert parser.scripts == [
        {"id": "x' onerror='alert(1)", "type": "application/json"}
    ]
    embedded = content.split("type='application/json'>", 1)[1].split("</script>")[0]
    assert json.loads(embedded) == data
    assert "<" not in embedded and "&" not in embedded
    assert "\u2028" not in embedded and "\u2029" not in embedded
    assert attack not in content
    assert "default-src 'none'; script-src 'none';" in content
    assert "<section><h1>確認設定</h1></section>" in content


def test_document_enables_only_the_supplied_builtin_script() -> None:
    rendered = render_html_document(
        title="報告", body="<p>內容</p>", script="'use strict';", style="p{color:red}"
    )
    assert "script-src 'unsafe-inline'" in rendered
    assert "<script>'use strict';</script>" in rendered
    assert "p{color:red}</style>" in rendered
    assert "type='application/json'" not in rendered


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_document_refuses_nonfinite_embedded_evidence(value: float) -> None:
    with pytest.raises(ValueError, match="Out of range float"):
        render_html_document(title="確認", body="", data={"值": value})
