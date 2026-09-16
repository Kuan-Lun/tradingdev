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
    from collections.abc import Iterator

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
    target = f"strategy_id={scenario.strategy_id!r}"

    def require_index(description: str, indices: Iterator[int]) -> int:
        index = next(indices, None)
        if index is None:
            raise AssertionError(f"Missing required MCP step: {description}")
        return index

    contract_index = require_index(
        f"get_strategy_contract before saving {target}",
        (
            index
            for index, call in enumerate(calls)
            if call.name == "get_strategy_contract"
        ),
    )
    save_index = require_index(
        f"save_strategy({target}) returning success=True",
        (
            index
            for index, call in enumerate(calls)
            if call.name == "save_strategy"
            and call.arguments.get("strategy_id") == scenario.strategy_id
            and call.result
            and call.result.get("success")
        ),
    )
    assert contract_index < save_index, (
        f"Invalid MCP step order for {target}: get_strategy_contract must precede "
        f"the first successful save_strategy (events {contract_index}, {save_index})"
    )
    previous = save_index
    for name, status in (
        ("validate_strategy", "validated"),
        ("dry_run_strategy", "runnable"),
    ):
        previous = require_index(
            f"{name}({target}) returning success=True, status={status!r} "
            f"after {calls[previous].name} (event {previous})",
            (
                index
                for index, call in enumerate(calls)
                if index > previous
                and call.name == name
                and call.result
                and call.result.get("success")
                and call.result.get("status") == status
                and call.arguments.get("strategy_id") == scenario.strategy_id
            ),
        )
    start_index = require_index(
        f"start_backtest for {target} returning a nonempty job_id "
        "after successful dry_run_strategy",
        (
            index
            for index, call in enumerate(calls)
            if index > previous
            and call.name == "start_backtest"
            and call.result
            and call.result.get("job_id")
        ),
    )
    started = calls[start_index]
    expected_arguments = {
        "strategy_id": scenario.strategy_id,
        "symbol": "BTC/USDT",
        "timeframe": "1h",
        "start_date": "2024-01-01",
        "end_date": "2024-01-08",
    }
    assert started.arguments == expected_arguments, (
        f"start_backtest arguments differ for {target}: "
        f"expected {expected_arguments!r}; got {started.arguments!r}"
    )
    job_id = started.result["job_id"]
    done_index = require_index(
        f"get_job_status(job_id={job_id!r}) returning status='done' "
        f"after start_backtest for {target}",
        (
            index
            for index, call in enumerate(calls)
            if index > start_index
            and call.name == "get_job_status"
            and call.result
            and call.arguments.get("job_id") == job_id
            and call.result.get("status") == "done"
        ),
    )
    done = calls[done_index]
    run_id = done.result["run_id"]
    queried_index = require_index(
        f"get_run(run_id={run_id!r}) returning a result after "
        f"get_job_status reported done for {target}",
        (
            index
            for index, call in enumerate(calls)
            if index > done_index
            and call.name == "get_run"
            and call.result
            and call.arguments.get("run_id") == run_id
        ),
    )
    queried = calls[queried_index]
    assert queried.result["success"], (
        f"get_run(run_id={run_id!r}) did not return success=True: {queried.result!r}"
    )
    assert queried.result["run"]["metrics"] == done.result["metrics"], (
        f"get_run(run_id={run_id!r}) metrics differ from "
        f"get_job_status(job_id={job_id!r}) metrics"
    )
    assert queried.result["run"]["strategy_id"] == scenario.strategy_id, (
        f"get_run(run_id={run_id!r}) belongs to "
        f"strategy_id={queried.result['run']['strategy_id']!r}, expected {target}"
    )
    require_index(
        f"list_artifacts(run_id={run_id!r}) returning a nonempty result "
        f"after get_job_status reported done for {target}",
        (
            index
            for index, call in enumerate(calls)
            if index > done_index
            and call.name == "list_artifacts"
            and call.result
            and call.arguments.get("run_id") == run_id
        ),
    )
    if scenario.repair:
        read_index = require_index(
            f"get_strategy({target}) returning success=True before repairing the draft",
            (
                index
                for index, call in enumerate(calls)
                if call.name == "get_strategy"
                and call.arguments.get("strategy_id") == scenario.strategy_id
                and call.result
                and call.result.get("success")
            ),
        )
        failure_index = require_index(
            f"validate_strategy({target}) returning success=False with "
            "diagnostic code='invalid_signal_values' before repairing the draft",
            (
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
            ),
        )
        assert read_index < failure_index < save_index, (
            f"Invalid MCP repair order for {target}: successful get_strategy must "
            "precede validate_strategy with invalid_signal_values, which must "
            f"precede successful save_strategy (events {read_index}, "
            f"{failure_index}, {save_index})"
        )


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
