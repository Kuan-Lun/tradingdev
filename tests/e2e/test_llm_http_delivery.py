"""Natural remote reading requests through real HTTP MCP and a real model."""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import unquote

import httpx
import pytest
from markdown_it import MarkdownIt

from tests.app.test_diagnostics import analysis
from tests.app.test_report_service import _save
from tests.e2e.codex_harness import run_codex
from tests.e2e.conversation_user import _assert_summary, assert_undelivered_confirmation
from tests.e2e.llm_client import codex_calls, run_local_model
from tests.mcp.http_harness import http_plan_server
from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.adapters.storage.sqlite import SQLiteStore
from tradingdev.app.report_service import ReportService

if TYPE_CHECKING:
    from tests.e2e.conversation import ModelTurn
    from tests.e2e.llm_client import ToolCall

pytestmark = pytest.mark.live_llm


@pytest.mark.parametrize(
    "kind,http_access",
    [("confirmation", True), ("confirmation", False), ("report", True)],
)
def test_remote_http_reading_and_followup(
    monkeypatch: pytest.MonkeyPatch,
    pytestconfig: pytest.Config,
    kind: str,
    http_access: bool,
) -> None:
    """Loopback represents the client's forwarded endpoint, not remote DNS/TLS.

    The user states whether their browser can reach it; no tool names, order or
    correct response are provided. Without forwarding, a server loopback URL is
    correctly undeliverable even when the server successfully issues it.
    """

    async def check() -> None:
        async with (
            http_plan_server(monkeypatch) as runtime,
            runtime.connect() as client,
        ):
            workspace = WorkspacePaths(runtime.root / "workspace")
            store = SQLiteStore(workspace)
            if kind == "confirmation":
                prepared = await client.call(
                    "prepare_backtest", **runtime.context.arguments
                )
                descriptor = prepared["confirmation_artifact"]
                source = Path(descriptor["server_path"])
                artifact_id, sha = descriptor["artifact_id"], descriptor["sha256"]
                request = (
                    f"請重新取得準備好的計畫 {prepared['plan_id']}，"
                    "讓我閱讀完整確認單與設定摘要。"
                    "我還沒有同意執行。"
                )
            else:
                reports = ReportService(workspace=workspace, store=store)
                _save((workspace, store, reports))
                report = reports.generate_report(
                    ["run"], sections=[], analysis=analysis()
                )
                assert report["success"]
                source = Path(report["path"])
                artifact_id, sha = report["artifact_id"], report["sha256"]
                request = (
                    f"請重新取得已保存報告 {report['report_id']} 的閱讀連結，"
                    "讓我打開已經做好的那份報告。不要重做報告或重跑研究。"
                )
            request += (
                "我在另一台電腦，無法存取伺服器上的檔案；"
                "客戶端沒有附件或本機檔案寫入功能。"
            )
            request += (
                "客戶端已把服務的 HTTP 入口轉送到本機，瀏覽器也能使用這個轉送入口。"
                if http_access
                else "我沒有設定本機連接埠轉送，不能使用伺服器的 loopback 位址。"
            )
            turns: list[ModelTurn] = []

            async def user(turn: ModelTurn) -> str | None:
                turns.append(turn)
                print(f"[HTTP {kind}] Reply {len(turns)}:\n{turn.text}", flush=True)
                assert str(workspace.root) not in unquote(turn.text), turn.text
                assert not runtime.context.service.jobs.list_all_jobs()
                assert not any(
                    c.name.startswith("prepare_")
                    or (
                        c.name == "submit_execution_decision"
                        and c.arguments.get("decision") == "approve"
                    )
                    for c in turn.calls
                )
                if len(turns) == 1:
                    if not http_access:
                        assert_undelivered_confirmation(
                            turn.text, prepared["confirmation_summary"]
                        )
                        return "我決定不執行這份計畫，請取消它。"
                    deliveries = [
                        c.result
                        for c in turn.calls
                        if c.name == "get_artifact_download"
                        and c.result.get("success")
                        and c.result.get("artifact_id") == artifact_id
                    ]
                    assert deliveries, turn.text
                    result = deliveries[-1]
                    assert result["sha256"] == sha
                    await verify_link(turn.text, result)
                    if kind == "confirmation":
                        _assert_summary(prepared["confirmation_summary"], turn.text)
                        return "我決定不執行這份計畫，請取消它。"
                    return (
                        "請把這份已保存報告連同本次需求摘要打包給我，"
                        "方便離線閱讀和分享；不要執行策略。"
                    )
                if kind == "report":
                    exports = [
                        c.result
                        for c in turn.calls
                        if c.name == "export_research_note" and c.result.get("success")
                    ]
                    assert exports, turn.text
                    content = await verify_link(turn.text, exports[-1])
                    with zipfile.ZipFile(io.BytesIO(content)) as archive:
                        attachments = json.loads(archive.read("manifest.json"))[
                            "attachments"
                        ]
                        assert len(attachments) == 1
                        assert attachments[0]["artifact_id"] == artifact_id
                        assert (
                            archive.read(attachments[0]["path"]) == source.read_bytes()
                        )
                    assert "index.html" in turn.text and any(
                        word in turn.text for word in ("解壓", "解開")
                    )
                return None

            async def verify_link(text: str, result: dict[str, Any]) -> bytes:
                links = [
                    token.attrGet("href")
                    for block in MarkdownIt("commonmark").parse(text)
                    for token in block.children or []
                    if token.type == "link_open"
                ]
                assert links == [result["download_url"]], text
                async with httpx.AsyncClient() as reader:
                    response = await reader.get(result["download_url"])
                    assert response.status_code == 200
                    assert (
                        hashlib.sha256(response.content).hexdigest() == result["sha256"]
                    )
                assert any(
                    word in text for word in ("15", "十五", "期限", "過期", "到期")
                ), text
                return response.content

            names = [tool.name for tool in (await client.session.list_tools()).tools]
            assert all(name not in request for name in names)
            options: dict[str, Any] = {
                "model": pytestconfig.getoption("llm_model"),
                "reasoning_effort": pytestconfig.getoption("llm_reasoning_effort"),
                "timeout_seconds": pytestconfig.getoption("llm_timeout"),
                "user_reply": user,
                "tool_names": names,
            }
            calls: list[ToolCall]
            if pytestconfig.getoption("llm_provider") == "local":
                calls = await run_local_model(
                    client,
                    request,
                    base_url=pytestconfig.getoption("llm_base_url"),
                    temperature=pytestconfig.getoption("llm_temperature"),
                    **options,
                )
            else:
                host, port = runtime.address
                calls = codex_calls(
                    await run_codex(
                        runtime.root,
                        request,
                        mcp_url=f"http://{host}:{port}/mcp",
                        **options,
                    )
                )
            assert len(turns) == 2
            assert not any(c.name == "generate_report" for c in calls)
            assert not runtime.context.service.jobs.list_all_jobs()
            if kind == "confirmation":
                decisions = [c for c in calls if c.name == "submit_execution_decision"]
                assert len(decisions) == 1 and decisions[0].arguments == {
                    "plan_id": prepared["plan_id"],
                    "decision": "decline",
                }
                assert decisions[0].result["code"] == "confirmation_declined"
            else:
                assert len(store.list_runs()) == 1

    asyncio.run(check())
