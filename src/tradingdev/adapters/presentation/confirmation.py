"""Text and offline HTML views of one authoritative confirmation document."""

from __future__ import annotations

from typing import TYPE_CHECKING

from tradingdev.adapters.presentation.html import escape, render_html_document

if TYPE_CHECKING:
    from tradingdev.domain.presentation.confirmation import (
        ConfirmationDocument,
        ConfirmationField,
    )


def _value(field: ConfirmationField) -> str:
    return field.value_text + (f" {field.unit}" if field.unit else "")


def _field_text(field: ConfirmationField) -> str:
    text = f"- {field.label}：{_value(field)}"
    if field.description:
        text += "。" + field.description
    if field.explanation_source == "model":
        text += "（名稱、說明與單位由模型提供）"
    return text


def render_confirmation_text(document: ConfirmationDocument) -> str:
    """Keep all settings and evidence in text while omitting technical identity."""
    lines = ["回測前確認", document.title, "策略說明（模型提供）：" + document.summary]
    for section in document.sections:
        lines.extend(("", section.title))
        lines.extend(section.paragraphs)
        lines.extend(_field_text(field) for field in section.fields)
    return "\n".join(lines) + "\n"


def _field_html(field: ConfirmationField) -> str:
    description = escape(field.description)
    if field.explanation_source == "model":
        description += "<small>（名稱、說明與單位由模型提供）</small>"
    return (
        f"<tr><th>{escape(field.label)}</th><td>{escape(_value(field))}</td>"
        f"<td>{description}</td></tr>"
    )


def _table(fields: tuple[ConfirmationField, ...]) -> str:
    return (
        "<table><thead><tr><th>設定</th><th>內容</th><th>說明</th></tr></thead>"
        "<tbody>" + "".join(_field_html(field) for field in fields) + "</tbody></table>"
    )


def render_confirmation_html(document: ConfirmationDocument) -> str:
    """Render the same complete template; detailed execution settings stay visible."""
    body = (
        "<header><span class='badge'>回測前確認</span>"
        f"<h1>{escape(document.title)}</h1>"
        "<p class='muted'>策略名稱與下方說明由模型提供。</p>"
        f"<p>{escape(document.summary)}</p></header>"
    )
    for section in document.sections:
        body += (
            f"<section><h2>{escape(section.title)}</h2>"
            + "".join(f"<p>{escape(text)}</p>" for text in section.paragraphs)
            + (_table(section.fields) if section.fields else "")
            + "</section>"
        )
    identity = document.identity
    body += (
        "<details class='panel'><summary>技術識別與檔案資訊</summary>"
        "<p>這些資訊用於確認執行版本，不需要修改檔案或設定。</p>"
        f"<p>執行內容摘要：<code>{escape(identity.manifest_hash)}</code></p>"
        + (
            f"<p>確認計畫：<code>{escape(identity.plan_id)}</code></p>"
            if identity.plan_id
            else ""
        )
        + _table(identity.details)
        + "</details>"
    )
    return render_html_document(title="TradingDev · 回測前確認", body=body)
