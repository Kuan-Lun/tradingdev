"""A real Codex model authors and validates a strategy through MCP."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from tests.e2e.codex_harness import (
    run_codex,
    successful_mcp_calls,
    verify_generated_strategy,
)

pytestmark = pytest.mark.live_llm

_PROMPT = """\
請透過 TradingDev MCP，為我開發一個名為 codex_sma_integration 的策略。
需求：用收盤價的 5 根與 20 根簡單移動平均線判斷方向。當短均線高於長均線，
每根 bar 的 signal 都是 1；低於時每根都是 -1；相等或不足 20 根時是 0。
參數 fast_period=5、slow_period=20 必須放在 YAML strategy.parameters，並能覆寫。
BTC/USDT、1h、2024-01-01 至 2024-12-31，初始資金 10000，採用 signal 回測模式。
遵循 MCP 提供的策略合約，將草稿儲存至後端，修正任何驗證錯誤，直到
validate_strategy 與 dry_run_strategy 都成功，策略狀態成為 runnable。
只使用 TradingDev MCP 工具完成程式碼與 YAML 的儲存、檢查與修正，不使用 shell
或檔案編輯工具。不下載行情、不啟動回測、不 promote。完成後簡短報告策略狀態。
"""


def test_codex_authors_runnable_strategy_through_mcp() -> None:
    with TemporaryDirectory(prefix="tradingdev-codex-e2e-") as temporary:
        root = Path(temporary)
        events = asyncio.run(
            run_codex(
                root,
                _PROMPT,
                timeout_seconds=float(
                    os.environ.get("TRADINGDEV_CODEX_TIMEOUT", "300")
                ),
            )
        )
        calls = successful_mcp_calls(events)
        tools = [call["tool"] for call in calls]
        required = [
            "list_strategies",
            "get_strategy_contract",
            "save_strategy",
            "validate_strategy",
            "dry_run_strategy",
        ]
        for tool in required:
            assert tool in tools, f"Codex never completed MCP {tool}; called {tools}"
        assert tools.index("get_strategy_contract") < tools.index("save_strategy")
        assert tools.index("save_strategy") < tools.index("validate_strategy")
        assert tools.index("validate_strategy") < tools.index("dry_run_strategy")

        verify_generated_strategy(root)
    assert not root.exists(), "The live Codex test left temporary artifacts behind"
