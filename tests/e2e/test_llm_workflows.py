"""Explicit model tests: author/repair, run a real backtest, and read its result."""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any

import pytest
import yaml

from tests.e2e.codex_harness import run_codex, verify_generated_strategy
from tests.e2e.conversation_user import ObservedPlanUser
from tests.e2e.llm_client import ToolCall, codex_calls, run_local_model
from tests.e2e.report_delivery import verify_report_delivery
from tests.e2e.strategy_scenarios import SCENARIOS, Scenario, market_frame
from tests.e2e.workflow_diagnostics import workflow_diagnostics
from tests.e2e.workflow_evidence import DETAIL_METRIC_IDS, assert_workflow
from tests.integration.mcp_harness import temporary_mcp_workspace
from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.domain.presentation.reports import REPORT_RECIPES

if TYPE_CHECKING:
    from pathlib import Path

    from tests.integration.mcp_harness import MCPWorkspace

pytestmark = pytest.mark.live_llm


async def seed_broken_draft(workspace: MCPWorkspace, scenario: Scenario) -> None:
    async with workspace.connect() as client:
        contract = (await client.call("get_strategy_contract", section="example"))[
            "content"
        ]
        if scenario.runtime_exit:
            code = contract["example_strategy_code"].replace(
                "        result = df.copy()",
                "        if len(df):\n            raise SystemExit(7)\n"
                "        result = df.copy()",
            )
            assert "raise SystemExit(7)" in code
        else:
            code = contract["example_strategy_code"].replace(
                "        return result",
                '        result["signal"] = np.nan\n        return result',
            )
            assert 'result["signal"] = np.nan' in code
        config = yaml.safe_load(contract["example_yaml_config"])
        config["strategy"]["parameters"] = scenario.parameters
        config["backtest"].update(
            start_date="2024-01-01",
            end_date="2024-01-08",
            init_cash=10000,
            fees=0,
            slippage=0,
            periods_per_year=365,
        )
        config["random_seed"] = 42
        saved = await client.call(
            "save_strategy",
            strategy_id=scenario.strategy_id,
            code=code,
            yaml_config=yaml.safe_dump(config),
            research=scenario.research,
        )
        assert saved["success"] and saved["status"] == "draft", saved


async def seed_legacy_strategy(
    workspace: MCPWorkspace, scenario: Scenario
) -> dict[Path, bytes]:
    """Create untouched pre-revision files solely inside the test workspace."""
    async with workspace.connect() as client:
        contract = (await client.call("get_strategy_contract", section="example"))[
            "content"
        ]
    generated = workspace.workspace / "generated_strategies"
    configs = workspace.workspace / "configs"
    generated.mkdir(parents=True, exist_ok=True)
    configs.mkdir(parents=True, exist_ok=True)
    source = generated / f"{scenario.legacy_strategy_id}.py"
    config_path = configs / f"{scenario.legacy_strategy_id}.yaml"
    config = yaml.safe_load(contract["example_yaml_config"])
    config["strategy"].update(
        id=scenario.legacy_strategy_id,
        parameters=scenario.parameters,
        source_path=str(source),
    )
    metadata = {
        "strategy_id": scenario.legacy_strategy_id,
        "class_name": config["strategy"]["class_name"],
        "status": "runnable",
        "source_path": str(source),
        "config_path": str(config_path),
        "validation": {"success": True},
        "dry_run": {"success": True},
    }
    originals = {
        source: contract["example_strategy_code"].encode("utf-8"),
        config_path: yaml.safe_dump(config).encode("utf-8"),
        generated / f"{scenario.legacy_strategy_id}.json": json.dumps(metadata).encode(
            "utf-8"
        ),
    }
    for path, content in originals.items():
        path.write_bytes(content)
    return originals


@pytest.mark.parametrize("scenario", SCENARIOS.values(), ids=SCENARIOS.keys())
def test_llm_authors_backtests_and_queries_results(
    pytestconfig: pytest.Config, scenario: Scenario
) -> None:
    provider = pytestconfig.getoption("llm_provider")
    assert provider in {"codex", "local"}, "Use scripts/check-llm.sh codex|local"
    with (
        temporary_mcp_workspace() as workspace,
        workflow_diagnostics(workspace, scenario),
    ):
        workspace.seed_market(market_frame())
        legacy_files: dict[Path, bytes] = {}
        user = ObservedPlanUser(workspace)

        async def run() -> list[ToolCall]:
            if scenario.repair:
                await seed_broken_draft(workspace, scenario)
            if scenario.legacy:
                legacy_files.update(await seed_legacy_strategy(workspace, scenario))
            options: dict[str, Any] = {
                "timeout_seconds": pytestconfig.getoption("llm_timeout"),
                "model": pytestconfig.getoption("llm_model"),
                "reasoning_effort": pytestconfig.getoption("llm_reasoning_effort"),
            }
            if scenario.runtime_exit:
                async with workspace.connect() as client:
                    names = [
                        tool.name for tool in (await client.session.list_tools()).tools
                    ]
                    assert all(name not in scenario.prompt for name in names)
                    options["tool_names"] = names
            if provider == "codex":
                calls = codex_calls(
                    await run_codex(
                        workspace.root,
                        scenario.prompt,
                        user_reply=user,
                        **options,
                    )
                )
            else:
                async with workspace.connect() as client:
                    calls = await run_local_model(
                        client,
                        scenario.prompt,
                        base_url=pytestconfig.getoption("llm_base_url"),
                        temperature=pytestconfig.getoption("llm_temperature"),
                        user_reply=user,
                        **options,
                    )
            user.assert_complete()
            return calls

        calls = asyncio.run(run())
        try:
            assert_workflow(calls, scenario)
            detail = next(
                call for call in reversed(calls) if call.name == "get_run_metrics"
            )
            saved = json.loads(
                (
                    workspace.workspace
                    / "runs"
                    / detail.arguments["run_id"]
                    / "performance.json"
                ).read_text(encoding="utf-8")
            )
            persisted = saved["scopes"][detail.result["scope"]]["values"]
            for metric_id in DETAIL_METRIC_IDS:
                assert detail.result["metrics"][metric_id] == persisted[metric_id]
            if scenario.runtime_exit:
                verify_report_delivery(
                    WorkspacePaths(workspace.workspace),
                    user.turns[-1],
                    detail.arguments["run_id"],
                    expected_sections=list(REPORT_RECIPES["standard"]),
                    expected_commentary=None,
                    required_analysis_refs={(detail.arguments["run_id"], "full")},
                )
                print(
                    "[runtime-repair] Final model reply:\n" + user.turns[-1].text,
                    flush=True,
                )
            if scenario.history:
                print(
                    "[history] Final model reply (test evidence; temporary links "
                    "expire at cleanup):\n" + user.turns[-1].text,
                    flush=True,
                )
                report_html = verify_report_delivery(
                    WorkspacePaths(workspace.workspace),
                    user.turns[-1],
                    detail.arguments["run_id"],
                    expected_sections=["overview", "metrics", "ledger", "trades"],
                    expected_commentary=[
                        {"title": "研究評語", "text": "本次結果僅為歷史模擬。"}
                    ],
                )
                assert detail.arguments["run_id"] in report_html
                assert "<table" in report_html
                assert "本次結果僅為歷史模擬。" in report_html
                observations = json.loads(
                    (
                        workspace.workspace
                        / "runs"
                        / detail.arguments["run_id"]
                        / "observations.json"
                    ).read_text(encoding="utf-8")
                )["scopes"]["full"]
                for call in calls:
                    if call.name == "get_run_executions" and call.result.get("success"):
                        assert (
                            call.result["records"]
                            == observations["execution_records"][:1]
                        )
                    if call.name == "get_run_account_history" and call.result.get(
                        "success"
                    ):
                        assert (
                            call.result["states"] == observations["account_history"][:2]
                        )
                    if call.name == "get_run_trades" and call.result.get("success"):
                        for trade in call.result["trades"]:
                            assert (
                                trade["record"]
                                == observations["trades"][trade["trade_id"]]
                            )
            for path, original in legacy_files.items():
                assert path.read_bytes() == original, (
                    f"Legacy recovery modified the original file: {path.name}"
                )
        except (AssertionError, StopIteration, ValueError) as exc:
            pytest.fail(
                f"Incomplete {provider}/{scenario.name} workflow: {exc}\n{calls!r}"
            )
        verify_generated_strategy(workspace.root, scenario=scenario.name)
