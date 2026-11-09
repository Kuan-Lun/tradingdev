"""Real model/MCP discovery and rejected preflights; no market fetch or training.

Read-only discovery uses actual product contracts. Unsupported preparation modes
reach production workers, which reject them before fetching data or training.
No worker or service is substituted, and callbacks never grant execution approval.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict
from typing import TYPE_CHECKING, Any

import pytest

from tests.e2e.codex_harness import run_codex
from tests.e2e.discovery_workflow import verify_readonly_attempts
from tests.e2e.llm_client import ToolCall, codex_calls, run_local_model
from tests.e2e.report_delivery import verify_no_execution_attempts
from tests.e2e.strategy_execution_guidance import (
    GUIDANCE_PROMPT,
    GUIDANCE_TOOLS,
    assert_guidance_workflow,
    assert_guidance_workspace,
    seed_guidance_evidence,
)
from tests.integration.mcp_harness import temporary_mcp_workspace

if TYPE_CHECKING:
    from tests.e2e.conversation import ModelTurn


@pytest.mark.live_llm
def test_natural_generated_runtime_boundary_guidance(
    pytestconfig: pytest.Config,
) -> None:
    """Find the isolation contract from user intent without naming its tool."""
    with temporary_mcp_workspace() as workspace:
        turns: list[ModelTurn] = []
        readonly_tools: set[str] = set()
        request = (
            "我想在自己的 Mac 上使用助手寫的新策略。它會直接在 Mac 的 Python 裡跑嗎？"
            "能讀我的檔案、連網或任意建立子程序嗎？若關掉視窗或取消工作又會怎樣？"
            "若顯示清理未確認，我應該怎麼處理？"
            "請查目前產品契約，只說明使用條件、資源上限如何設定與必要限制，"
            "不要建立策略或執行研究。"
        )

        async def observe(turn: ModelTurn) -> None:
            turns.append(turn)

        async def run() -> list[ToolCall]:
            options: dict[str, Any] = {
                "model": pytestconfig.getoption("llm_model"),
                "reasoning_effort": pytestconfig.getoption("llm_reasoning_effort"),
                "timeout_seconds": pytestconfig.getoption("llm_timeout"),
                "user_reply": observe,
                "max_turns": 1,
            }
            async with workspace.connect() as client:
                tools = (await client.session.list_tools()).tools
                assert all(tool.name not in request for tool in tools)
                options["tool_names"] = [tool.name for tool in tools]
                readonly_tools.update(
                    tool.name
                    for tool in tools
                    if tool.annotations and tool.annotations.read_only_hint is True
                )
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
        verify_readonly_attempts(calls, readonly_tools)
        assert any(
            call.name == "get_strategy_contract"
            and call.result.get("section") == "runtime"
            and call.result.get("content", {}).get("limits_schema")
            for call in calls
        ), calls
        assert turns and turns[-1].text.strip()
        print("[runtime-boundary] Final model reply:\n" + turns[-1].text, flush=True)
    assert not workspace.root.exists()


@pytest.mark.live_llm
def test_llm_terminal_cash_explanation(pytestconfig: pytest.Config) -> None:
    """Discover end-of-run cash semantics without tool names or supplied answers."""
    with temporary_mcp_workspace() as workspace:
        turns: list[ModelTurn] = []
        readonly_tools: set[str] = set()

        async def user(turn: ModelTurn) -> str | None:
            turns.append(turn)
            if len(turns) == 1:
                return (
                    "如果最後一根開盤才買進，或在那時由多翻空，結果又怎麼計算？"
                    "假如空倉已經虧到無法買回呢？重建舊回測的報告會改掉舊數值嗎？"
                    "仍只解釋，不執行研究。"
                )
            return None

        request = (
            "我想從投資人的角度解讀 TradingDev 回測結束時可拿回多少錢。"
            "請查目前規則：Signal 回測到最後還持有部位時，結果是帳面市值還是現金？"
            "進場和出場成本會怎麼算？Volume 模式及買進持有對照也一樣嗎？"
            "只查詢與解釋，不建立策略或執行研究。"
        )

        async def run() -> list[ToolCall]:
            options: dict[str, Any] = {
                "model": pytestconfig.getoption("llm_model"),
                "reasoning_effort": pytestconfig.getoption("llm_reasoning_effort"),
                "timeout_seconds": pytestconfig.getoption("llm_timeout"),
                "user_reply": user,
                "max_turns": 2,
            }
            async with workspace.connect() as client:
                tools = (await client.session.list_tools()).tools
                options["tool_names"] = [tool.name for tool in tools]
                assert all(tool.name not in request for tool in tools)
                readonly_tools.update(
                    tool.name
                    for tool in tools
                    if tool.annotations and tool.annotations.read_only_hint is True
                )
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
        verify_readonly_attempts(calls, readonly_tools)
        assert len(turns) == 2 and all(turn.text.strip() for turn in turns)
        assert any(
            call.name == "get_strategy_contract"
            and call.result.get("section") == "configuration"
            and "liquidates remaining positions" in json.dumps(call.result)
            for call in turns[0].calls
        ), "The explanation must use the current product contract"
        for index, turn in enumerate(turns, 1):
            print(f"[terminal cash] Semantic review, turn {index}:\n{turn.text}")
    assert not workspace.root.exists()


@pytest.mark.live_llm
def test_llm_glft_time_units(pytestconfig: pytest.Config) -> None:
    """Discover both bundled contracts and explain frequency changes naturally."""
    with temporary_mcp_workspace() as workspace:
        turns: list[ModelTurn] = []
        readonly_tools: set[str] = set()

        async def user(turn: ModelTurn) -> str | None:
            turns.append(turn)
            if len(turns) == 1:
                return (
                    "如果原始行情中間少了一根，或者從 00:02 開始，會怎麼處理？"
                    "用分鐘資料訓練過的模型能直接換成小時資料嗎？仍只解釋，不執行研究。"
                )
            return None

        request = (
            "請查目前內建 GLFT 與 GLFT-ML 的實作設定，說明把分鐘 K 換成小時 K 後，"
            "波動率、EMA 聚合、持倉根數與 ML 預測期間如何解讀。"
            "以年化 DVOL=50 為例比較兩種頻率的每根波動率。"
            "也說明以前用分鐘命名的聚合設定現在怎麼設定。只查詢與解釋，不建立或執行研究。"
        )

        async def run() -> list[ToolCall]:
            options: dict[str, Any] = {
                "model": pytestconfig.getoption("llm_model"),
                "reasoning_effort": pytestconfig.getoption("llm_reasoning_effort"),
                "timeout_seconds": pytestconfig.getoption("llm_timeout"),
                "user_reply": user,
                "max_turns": 2,
            }
            async with workspace.connect() as client:
                tools = (await client.session.list_tools()).tools
                options["tool_names"] = [tool.name for tool in tools]
                readonly_tools.update(
                    tool.name
                    for tool in tools
                    if tool.annotations and tool.annotations.read_only_hint is True
                )
                assert all(tool.name not in request for tool in tools)
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
        verify_readonly_attempts(calls, readonly_tools)
        retrieved = {
            call.arguments.get("strategy_id")
            for call in calls
            if call.name == "get_strategy" and call.result.get("success")
        }
        assert {"glft_market_making", "glft_ml"} <= retrieved
        assert len(turns) == 2 and all(turn.text.strip() for turn in turns)
        for index, turn in enumerate(turns, 1):
            print(f"[GLFT time units] Semantic review, turn {index}:\n{turn.text}")
    assert not workspace.root.exists()


@pytest.mark.live_llm
def test_llm_strategy_execution_guidance(pytestconfig: pytest.Config) -> None:
    provider = pytestconfig.getoption("llm_provider")
    assert provider in {"codex", "local"}, "Use scripts/check-llm.sh codex|local"
    with temporary_mcp_workspace() as workspace:
        seed_guidance_evidence(workspace)
        turns: list[ModelTurn] = []

        async def observe_final(turn: ModelTurn) -> None:
            turns.append(turn)

        async def run() -> list[ToolCall]:
            options: dict[str, Any] = {
                "timeout_seconds": pytestconfig.getoption("llm_timeout"),
                "model": pytestconfig.getoption("llm_model"),
                "reasoning_effort": pytestconfig.getoption("llm_reasoning_effort"),
                "tool_names": GUIDANCE_TOOLS,
                "user_reply": observe_final,
                "max_tool_calls": 8,
                "max_turns": 1,
            }
            if provider == "codex":
                return codex_calls(
                    await run_codex(workspace.root, GUIDANCE_PROMPT, **options)
                )
            async with workspace.connect() as client:
                return await run_local_model(
                    client,
                    GUIDANCE_PROMPT,
                    base_url=pytestconfig.getoption("llm_base_url"),
                    temperature=pytestconfig.getoption("llm_temperature"),
                    **options,
                )

        calls = asyncio.run(run())
        try:
            assert_guidance_workflow(calls, turns)
            assert_guidance_workspace(workspace)
        except AssertionError as error:
            transcript = {
                "calls": [asdict(call) for call in calls],
                "turns": [asdict(turn) for turn in turns],
            }
            pytest.fail(
                f"Incomplete {provider}/strategy execution guidance: {error}\n"
                + json.dumps(transcript, ensure_ascii=False)
            )
    assert not workspace.root.exists()


@pytest.mark.live_llm
def test_llm_volume_monthly_loss_explanation(pytestconfig: pytest.Config) -> None:
    """Discover current policy from natural questions, including a follow-up."""
    with temporary_mcp_workspace() as workspace:
        turns: list[ModelTurn] = []

        async def user(turn: ModelTurn) -> str | None:
            turns.append(turn)
            if len(turns) == 1:
                return (
                    "若一月進場時已扣過手續費，跨月平倉時還會算進本月限制嗎？"
                    "如果二、三月都沒有行情，到四月才出現下一根 K 棒又怎麼算？"
                    "請沿用剛才查到的規則說明，不執行任何研究。"
                )
            return None

        request = (
            "請查明 TradingDev 的 Volume 模式如何計算每月虧損限制。"
            "假設一月買進後跨月持有，月底累積損益為 20，二月第一根收盤變成 10，"
            "每月最大虧損設 5，零手續費及滑價。會繼續持有嗎？"
            "交易明細應保留哪一段損益？只解釋目前規則，不建立或執行策略。"
        )

        async def run() -> list[ToolCall]:
            options: dict[str, Any] = {
                "model": pytestconfig.getoption("llm_model"),
                "reasoning_effort": pytestconfig.getoption("llm_reasoning_effort"),
                "timeout_seconds": pytestconfig.getoption("llm_timeout"),
                "user_reply": user,
                "max_turns": 2,
            }
            async with workspace.connect() as client:
                options["tool_names"] = [
                    tool.name for tool in (await client.session.list_tools()).tools
                ]
                assert all(name not in request for name in options["tool_names"])
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
        verify_no_execution_attempts(calls)
        assert len(turns) == 2 and all(turn.text.strip() for turn in turns)
        for index, turn in enumerate(turns, 1):
            print(
                f"[monthly loss] Semantic review, turn {index}:\n{turn.text}",
                flush=True,
            )
        assert any(
            call.name == "get_strategy_contract"
            and call.result.get("section") == "configuration"
            and "previous observed month's last close equity" in json.dumps(call.result)
            and "Trade records retain whole-trade P&L and costs"
            in json.dumps(call.result)
            for call in turns[0].calls
        ), "The first explanation must be grounded in the current execution contract"
    assert not workspace.root.exists()
