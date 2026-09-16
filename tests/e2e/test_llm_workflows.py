"""Explicit model tests: author/repair, run a real backtest, and read its result."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import pytest
import yaml

from tests.e2e.codex_harness import run_codex, verify_generated_strategy
from tests.e2e.llm_client import ToolCall, codex_calls, run_local_model
from tests.e2e.strategy_scenarios import SCENARIOS, Scenario, market_frame
from tests.integration.mcp_harness import temporary_mcp_workspace

if TYPE_CHECKING:
    from tests.integration.mcp_harness import MCPWorkspace

pytestmark = pytest.mark.live_llm


async def seed_broken_draft(workspace: MCPWorkspace, scenario: Scenario) -> None:
    async with workspace.connect() as client:
        contract = await client.call("get_strategy_contract")
        code = contract["example_strategy_code"].replace(
            'result["signal"] = 0', 'result["signal"] = 7'
        )
        assert 'result["signal"] = 7' in code
        config = yaml.safe_load(contract["example_yaml_config"])
        config["strategy"]["parameters"] = scenario.parameters
        saved = await client.call(
            "save_strategy",
            strategy_id=scenario.strategy_id,
            code=code,
            yaml_config=yaml.safe_dump(config),
        )
        assert saved["success"] and saved["status"] == "draft", saved


def assert_workflow(calls: list[ToolCall], scenario: Scenario) -> None:
    """Require actual successful tool results in order, not a model's summary."""
    names = [call.name for call in calls]
    contract_index = names.index("get_strategy_contract")
    save_index = next(
        index
        for index, call in enumerate(calls)
        if call.name == "save_strategy"
        and call.arguments.get("strategy_id") == scenario.strategy_id
        and call.result
        and call.result.get("success")
    )
    assert contract_index < save_index, names
    previous = save_index
    for name, status in (
        ("validate_strategy", "validated"),
        ("dry_run_strategy", "runnable"),
    ):
        previous = next(
            index
            for index, call in enumerate(calls)
            if index > previous
            and call.name == name
            and call.result
            and call.result.get("success")
            and call.result.get("status") == status
            and call.arguments.get("strategy_id") == scenario.strategy_id
        )
    start_index, started = next(
        (index, call)
        for index, call in enumerate(calls)
        if index > previous
        and call.name == "start_backtest"
        and call.result
        and call.result.get("job_id")
    )
    expected_arguments = {
        "strategy_id": scenario.strategy_id,
        "symbol": "BTC/USDT",
        "timeframe": "1h",
        "start_date": "2024-01-01",
        "end_date": "2024-01-08",
    }
    assert started.arguments == expected_arguments, started
    done_index, done = next(
        (index, call)
        for index, call in enumerate(calls)
        if index > start_index
        and call.name == "get_job_status"
        and call.result
        and call.arguments.get("job_id") == started.result["job_id"]
        and call.result.get("status") == "done"
    )
    queried = next(
        call
        for index, call in enumerate(calls)
        if index > done_index
        and call.name == "get_run"
        and call.result
        and call.arguments.get("run_id") == done.result["run_id"]
    )
    assert queried.result["success"], queried
    assert queried.result["run"]["metrics"] == done.result["metrics"]
    assert queried.result["run"]["strategy_id"] == scenario.strategy_id
    assert any(
        index > done_index
        and call.name == "list_artifacts"
        and call.result
        and call.arguments.get("run_id") == done.result["run_id"]
        for index, call in enumerate(calls)
    )
    if scenario.repair:
        read_index = next(
            index
            for index, call in enumerate(calls)
            if call.name == "get_strategy"
            and call.arguments.get("strategy_id") == scenario.strategy_id
            and call.result
            and call.result.get("success")
        )
        failure_index = next(
            index
            for index, call in enumerate(calls)
            if call.name == "validate_strategy"
            and call.arguments.get("strategy_id") == scenario.strategy_id
            and call.result
            and call.result.get("success") is False
            and any(
                diagnostic.get("code") == "invalid_signal_values"
                for diagnostic in call.result.get("diagnostics", [])
            )
        )
        assert read_index < failure_index < save_index


@pytest.mark.parametrize("scenario", SCENARIOS.values(), ids=SCENARIOS.keys())
def test_llm_authors_backtests_and_queries_results(
    pytestconfig: pytest.Config, scenario: Scenario
) -> None:
    provider = pytestconfig.getoption("llm_provider")
    assert provider in {"codex", "local"}, "Use scripts/check-llm.sh codex|local"
    with temporary_mcp_workspace() as workspace:
        workspace.seed_market(market_frame())

        async def run() -> list[ToolCall]:
            if scenario.repair:
                await seed_broken_draft(workspace, scenario)
            options: dict[str, Any] = {
                "timeout_seconds": pytestconfig.getoption("llm_timeout"),
                "model": pytestconfig.getoption("llm_model"),
            }
            if provider == "codex":
                return codex_calls(
                    await run_codex(workspace.root, scenario.prompt, **options)
                )
            async with workspace.connect() as client:
                return await run_local_model(
                    client,
                    scenario.prompt,
                    base_url=pytestconfig.getoption("llm_base_url"),
                    temperature=pytestconfig.getoption("llm_temperature"),
                    reasoning_effort=pytestconfig.getoption("llm_reasoning_effort"),
                    **options,
                )

        calls = asyncio.run(run())
        try:
            assert_workflow(calls, scenario)
        except (AssertionError, StopIteration, ValueError) as exc:
            pytest.fail(
                f"Incomplete {provider}/{scenario.name} workflow: {exc}\n{calls!r}"
            )
        verify_generated_strategy(workspace.root, scenario=scenario.name)
