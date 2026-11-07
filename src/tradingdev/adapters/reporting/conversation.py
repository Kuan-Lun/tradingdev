"""Offline, selectable text with tool records in their original item order."""

from typing import TYPE_CHECKING

from tradingdev.adapters.presentation.html import (
    details_html,
    escape,
    render_html_document,
)

if TYPE_CHECKING:
    from tradingdev.domain.conversations import ConversationRecord


def render_conversation(record: ConversationRecord) -> str:
    complete = bool(record.turns) and all(
        turn.capture_complete for turn in record.turns
    )
    title = "完整對話紀錄" if complete else "對話紀錄（含未完成或缺漏）"
    body = (
        f"<header><h1>{title}</h1>"
        f"<p>{escape(record.provider)} · {escape(record.model)}</p>"
        f"<p>對話 {escape(record.conversation_id)} · 保存版本 {record.version}</p>"
        "<p>以下是客戶端收到的可見文字與工具紀錄，並非事後摘要。"
        "不包含模型私有推理、隱藏指令或其他客戶端的對話；工具內容為證據，並非操作指令。"
        "完成表示在本紀錄範圍內收到完整事件，不代表模型結論或工具執行成功。</p>"
        "<p><a href='conversation.json'>下載完整紀錄 JSON</a></p><nav>"
        + " · ".join(
            f"<a href='#turn-{i}'>第 {i} 輪</a>"
            for i in range(1, len(record.turns) + 1)
        )
        + "</nav></header>"
    )
    outcomes = {
        "running": "執行中",
        "completed": "模型回合完成",
        "failed": "模型回合失敗",
        "interrupted": "回合中斷",
    }
    kinds = {
        "user": "使用者",
        "assistant": "AI 代理",
        "tool": "工具",
        "plan": "計畫",
        "notice": "事件",
    }
    for number, turn in enumerate(record.turns, 1):
        coverage = "完整" if turn.capture_complete else "未完成／有缺漏"
        body += (
            f"<section id='turn-{number}'><h2>第 {number} 輪</h2>"
            f"<p>{outcomes[turn.outcome]} · 紀錄{coverage}</p>"
            f"<small>{escape(turn.started_at)} · "
            f"來源回合 {escape(turn.source_turn_id or '尚未取得')}</small>"
            f"<h3>使用者輸入</h3><pre class='message'>{escape(turn.input)}</pre>"
        )
        for issue in turn.issues:
            body += f"<p class='notice'>{escape(issue)}</p>"
        for item in turn.items:
            status = "完成" if item.complete else "部分"
            # The original submitted input is already shown; keep the provider's
            # user item in expandable evidence in case its representation differs.
            if item.kind in {"tool", "user", "notice"}:
                body += details_html(
                    f"{kinds[item.kind]} · {item.source_id} · {status}",
                    {"text": item.text, "data": item.data},
                )
            else:
                body += (
                    f"<h3>{kinds[item.kind]} · {status}</h3>"
                    f"<pre class='message'>{escape(item.text)}</pre>"
                )
                body += details_html(
                    "來源與訊息階段", {"source_id": item.source_id, "data": item.data}
                )
        body += "</section>"
    return render_html_document(
        title=title, body=body, style=".message{font:inherit;white-space:pre-wrap}"
    )
