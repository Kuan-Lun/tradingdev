"""Natural interpretation of saved results, without a tool-by-tool script."""

from __future__ import annotations

import asyncio
import json
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from tests.app.test_report_service import _benchmark, _payload, _save, _scope
from tests.e2e.codex_harness import run_codex
from tests.e2e.conversation import links_to_file
from tests.e2e.conversation_user import assert_undelivered_confirmation
from tests.e2e.llm_client import codex_calls, run_local_model
from tests.e2e.report_delivery import (
    verify_archive_delivery,
    verify_no_execution_attempts,
    verify_report_delivery,
)
from tests.integration.mcp_harness import temporary_mcp_workspace
from tests.mcp.execution_fixtures import plan_context
from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.adapters.storage.sqlite import SQLiteStore
from tradingdev.app.contracts.plans import PlanResponse
from tradingdev.app.contracts.research_notes import NoteExport
from tradingdev.app.report_service import ReportService
from tradingdev.app.research_note_service import ResearchNoteService
from tradingdev.domain.preflight import PreflightRequest
from tradingdev.domain.presentation.reports import REPORT_RECIPES

pytestmark = pytest.mark.live_llm

if TYPE_CHECKING:
    from tests.e2e.conversation import ModelTurn
    from tests.e2e.llm_client import ToolCall


@pytest.mark.parametrize("copy_available", [True, False])
def test_remote_confirmation_delivery(
    pytestconfig: pytest.Config, copy_available: bool
) -> None:
    """Real model lookup/decline; a host-delivered copy or no file facility.

    Preparation uses substituted offline program checks. The client copy is real
    and hash-checked by the host fixture, not a model-authored or fake attachment.
    This exercises link selection, not a desktop client's attachment UI.
    """
    import hashlib
    from urllib.parse import quote, unquote

    with temporary_mcp_workspace() as workspace:
        context = plan_context(workspace.root)
        arguments = dict(context.arguments)
        evidence_id = arguments.pop("evidence_id")
        prepared = context.service.prepare(
            PreflightRequest(kind="backtest", arguments=arguments), evidence_id
        )
        assert isinstance(prepared, PlanResponse), prepared
        artifact = prepared.confirmation_artifact
        source = Path(artifact.server_path)
        delivered = workspace.root / "client 確認單 (副本).html"
        if copy_available:
            delivered.write_bytes(source.read_bytes())
            assert hashlib.sha256(delivered.read_bytes()).hexdigest() == artifact.sha256
            environment = (
                f"客戶端已把這份確認單完整下載到 {delivered}，"
                f"並核對 SHA-256 為 {artifact.sha256}，我能開啟這份本機副本。"
            )
        else:
            environment = "目前客戶端沒有檔案下載、附件或寫入本機檔案的功能。"
        request = (
            f"請重新取得剛才準備的計畫 {prepared.plan_id}，讓我閱讀確認單與設定摘要。"
            "我在另一台電腦使用，無法存取伺服器上的檔案。"
            f"{environment}我還沒有同意執行。"
        )
        turns: list[ModelTurn] = []

        async def user(turn: ModelTurn) -> str | None:
            turns.append(turn)
            assert source.read_bytes()
            assert str(source) not in unquote(turn.text), turn.text
            assert quote(str(source), safe="/") not in turn.text, turn.text
            assert not any(
                c.name == "submit_execution_decision"
                and c.arguments.get("decision") == "approve"
                for c in turn.calls
            )
            assert not context.service.jobs.list_all_jobs()
            if len(turns) == 1:
                assert any(
                    c.name == "get_execution_plan"
                    and c.arguments.get("plan_id") == prepared.plan_id
                    and c.result.get("success")
                    for c in turn.calls
                )
                if copy_available:
                    from tests.e2e.conversation_user import _assert_summary

                    _assert_summary(prepared.confirmation_summary, turn.text)
                    assert links_to_file(turn.text, delivered), turn.text
                else:
                    assert not delivered.exists()
                    assert not links_to_file(turn.text, delivered)
                    assert_undelivered_confirmation(
                        turn.text, prepared.confirmation_summary
                    )
                    assert any(word in turn.text for word in ("無法", "不能", "不支援"))
                print("[confirmation-delivery] First reply:\n" + turn.text, flush=True)
                return "我決定不執行這份計畫，請取消它。"
            return None

        async def run() -> list[ToolCall]:
            options: dict[str, Any] = {
                "model": pytestconfig.getoption("llm_model"),
                "reasoning_effort": pytestconfig.getoption("llm_reasoning_effort"),
                "timeout_seconds": pytestconfig.getoption("llm_timeout"),
                "user_reply": user,
            }
            async with workspace.connect() as client:
                names = [t.name for t in (await client.session.list_tools()).tools]
                assert all(name not in request for name in names)
                if pytestconfig.getoption("llm_provider") == "local":
                    return await run_local_model(
                        client,
                        request,
                        base_url=pytestconfig.getoption("llm_base_url"),
                        temperature=pytestconfig.getoption("llm_temperature"),
                        tool_names=names,
                        **options,
                    )
            return codex_calls(
                await run_codex(workspace.root, request, tool_names=names, **options)
            )

        calls = asyncio.run(run())
        assert len(turns) == 2
        decisions = [c for c in calls if c.name == "submit_execution_decision"]
        assert len(decisions) == 1
        assert decisions[0].arguments == {
            "plan_id": prepared.plan_id,
            "decision": "decline",
        }
        assert decisions[0].result["code"] == "confirmation_declined"
        assert not any(c.name.startswith("prepare_") for c in calls)
        assert not context.service.jobs.list_all_jobs()


def test_saved_research_interpretation(pytestconfig: pytest.Config) -> None:
    with temporary_mcp_workspace() as workspace:
        paths = WorkspacePaths(workspace.workspace)
        store = SQLiteStore(paths)
        scope = _scope(
            values={
                "total_return": 0.25,
                "max_drawdown": 0.1,
                "total_trades": 1,
                "win_rate": 1.0,
            }
        )
        execution_context = scope.metadata["execution_context"]
        assert isinstance(execution_context, dict)
        execution_context["fees"] = 0.0
        _save(
            (paths, store, ReportService(workspace=paths, store=store)),
            benchmarks={"full": _benchmark(fees=0.0, values={"total_return": 0.1})},
            scopes={"full": scope},
            trades=[
                {
                    "entry_idx": 0,
                    "exit_idx": 2,
                    "status": "closed",
                    "direction": 1,
                    "entry_price": 100.0,
                    "exit_price": 125.0,
                    "net_pnl": 25.0,
                }
            ],
        )
        turns: list[ModelTurn] = []

        async def user(turn: ModelTurn) -> str | None:
            turns.append(turn)
            if len(turns) == 1:
                html = verify_report_delivery(
                    paths,
                    turn,
                    "run",
                    expected_sections=list(REPORT_RECIPES["standard"]),
                    expected_commentary=None,
                    required_analysis_refs={("run", "full")},
                )
                assert "BTC/USDT 買進持有" in html
                delivered = next(
                    Path(c.result["path"])
                    for c in turn.calls
                    if c.name == "generate_report"
                    and c.result.get("success")
                    and links_to_file(turn.text, Path(c.result["path"]))
                )
                baseline = _payload(delivered)["runs"][0]["scopes"][0]["benchmark"]
                assert baseline["values"]["total_return"] == 0.1
                assert baseline["coverage"]["complete"] is True
                assert baseline["coverage"]["aligned_bars"] == 3
                assert baseline["coverage"]["scope_bars"] == 3
                return (
                    "請把這份研究目前的需求、討論摘要與這份報告存成研究筆記，"
                    "方便之後追加想法。不要重新回測。"
                )
            if len(turns) == 2:
                return (
                    "我追加一個後續研究想法：想評估改用 4 小時 K 線。"
                    "現在只記為未執行的需求，不要變更或執行策略。"
                    "請更新剛才的筆記，連同剛才的報告打包，給我能離線閱讀和分享的檔案。"
                )
            if len(turns) == 3:
                return (
                    "請重新取得剛才已保存的同一份封存，給我可閱讀及下載的資訊。"
                    "不要重建封存，也不要執行策略。"
                )
            return None

        request = (
            "請幫我解讀已保存的 BTC 策略回測，結果代號是 run。"
            "我要拿給同事討論：這次有哪些觀察、哪些事情還不能確定，"
            "也請和已保存的買進持有基準比較，交代基準資料覆蓋。"
            "下一步值得研究什麼？請給我能獨立閱讀的完整報告。不要重新回測。"
        )

        async def run() -> Any:
            options = {
                "model": pytestconfig.getoption("llm_model"),
                "reasoning_effort": pytestconfig.getoption("llm_reasoning_effort"),
                "timeout_seconds": pytestconfig.getoption("llm_timeout"),
                "user_reply": user,
            }
            async with workspace.connect() as client:
                options["tool_names"] = [
                    t.name for t in (await client.session.list_tools()).tools
                ]
                if pytestconfig.getoption("llm_provider") == "local":
                    return await run_local_model(
                        client,
                        request,
                        base_url=pytestconfig.getoption("llm_base_url"),
                        temperature=pytestconfig.getoption("llm_temperature"),
                        **options,
                    )
            return codex_calls(await run_codex(workspace.root, request, **options))

        calls = asyncio.run(run())
        assert turns
        verify_no_execution_attempts(calls)
        assert len(turns) == 4
        exports = [
            NoteExport.model_validate(c.result)
            for c in turns[2].calls
            if c.name == "export_research_note" and c.result.get("success")
        ]
        linked = [e for e in exports if links_to_file(turns[2].text, Path(e.path))]
        assert len(linked) == 1, "Final reply must link the actual ZIP"
        exported = linked[0]
        verify_archive_delivery(turns[2].text, exported)
        verify_archive_delivery(turns[-1].text, exported)
        reopened = [
            c
            for c in turns[-1].calls
            if c.name == "get_research_archive" and c.result.get("success")
        ]
        assert reopened and reopened[-1].result["archive_id"] == exported.archive_id
        assert not any(c.name == "export_research_note" for c in turns[-1].calls)
        notes = ResearchNoteService(paths, store)
        record = notes.notes.load(exported.note_id)
        assert record.version == exported.version and record.version >= 2
        assert len(notes.notes.list(0, 20)[0]) == 1, "Revise the same notebook"
        assert any(
            c.name == "get_research_notes" and c.result.get("success")
            for c in turns[2].calls
        )
        content = notes.archives.download(exported.archive_id, exported.sha256)
        assert content == Path(exported.path).read_bytes()
        with zipfile.ZipFile(exported.path) as archive:
            assert json.loads(archive.read("research.json")) == record.model_dump(
                mode="json"
            )
            manifest = json.loads(archive.read("manifest.json"))
            report_calls = [
                c
                for c in turns[0].calls
                if c.name == "generate_report" and c.result.get("success")
            ]
            delivered_ids = {
                c.result["artifact_id"]
                for c in report_calls
                if links_to_file(turns[0].text, Path(c.result["path"]))
            }
            assert delivered_ids <= {a["artifact_id"] for a in manifest["attachments"]}
            for attachment in manifest["attachments"]:
                registered = store.get_artifact(attachment["artifact_id"])
                assert registered is not None
                assert (
                    archive.read(attachment["path"])
                    == Path(registered["path"]).read_bytes()
                )
        print(
            "[research notebook] Semantic review: " + record.model_dump_json(),
            flush=True,
        )
        assert len(store.list_runs()) == 1
        print("[research delivery] Final reply:\n" + turns[-1].text, flush=True)
