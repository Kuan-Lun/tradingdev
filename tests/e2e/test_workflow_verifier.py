"""Offline regression checks for model-workflow evidence and semantic verifiers.

Each strategy runs once through real MCP and a real backtest worker. Subsequent
checks corrupt one piece of evidence at a time without launching another job.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
import yaml

from tests.e2e.codex_harness import verify_generated_strategy
from tests.e2e.llm_client import ToolCall
from tests.e2e.strategy_scenarios import SCENARIOS, Scenario, market_frame
from tests.e2e.test_llm_workflows import assert_workflow
from tests.integration.mcp_harness import MCPWorkspace, temporary_mcp_workspace

if TYPE_CHECKING:
    from collections.abc import Iterator

_SMA_CODE = '''\
"""Deterministic SMA direction fixture for workflow verification."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from tradingdev.domain import indicators
from tradingdev.domain.strategies.base import BaseStrategy

if TYPE_CHECKING:
    import pandas as pd


class SmaFixture(BaseStrategy):
    def __init__(self, fast_period: int, slow_period: int) -> None:
        self.fast_period = fast_period
        self.slow_period = slow_period

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        fast = indicators.sma(result["close"], self.fast_period)
        slow = indicators.sma(result["close"], self.slow_period)
        ready = np.isfinite(fast) & np.isfinite(slow)
        result["signal"] = 0
        result.loc[ready & (fast > slow), "signal"] = 1
        result.loc[ready & (fast < slow), "signal"] = -1
        return result

    def get_parameters(self) -> dict[str, Any]:
        return {
            "fast_period": self.fast_period,
            "slow_period": self.slow_period,
        }
'''

_MOMENTUM_CODE = '''\
"""Deterministic momentum fixture for workflow verification."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from tradingdev.domain.strategies.base import BaseStrategy

if TYPE_CHECKING:
    import pandas as pd


class MomentumFixture(BaseStrategy):
    def __init__(self, lookback: int) -> None:
        self.lookback = lookback

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        previous = result["close"].shift(self.lookback)
        result["signal"] = 0
        result.loc[result["close"] > previous, "signal"] = 1
        result.loc[result["close"] < previous, "signal"] = -1
        return result

    def get_parameters(self) -> dict[str, Any]:
        return {"lookback": self.lookback}
'''


@dataclass
class CompletedWorkflow:
    workspace: MCPWorkspace
    scenario: Scenario
    calls: list[ToolCall]
    source_path: Path
    config_path: Path
    run_id: str

    def verify(self) -> None:
        verify_generated_strategy(self.workspace.root, scenario=self.scenario.name)


async def _complete_workflow(
    workspace: MCPWorkspace, scenario: Scenario
) -> CompletedWorkflow:
    calls: list[ToolCall] = []
    async with workspace.connect() as client:

        async def call(name: str, **arguments: Any) -> Any:
            result = await client.call(name, **arguments)
            calls.append(ToolCall(name, arguments, result))
            return result

        contract = await call("get_strategy_contract")
        config = yaml.safe_load(contract["example_yaml_config"])
        if scenario.name == "momentum":
            code = _MOMENTUM_CODE
            config["strategy"]["class_name"] = "MomentumFixture"
        else:
            code = _SMA_CODE
            config["strategy"]["class_name"] = "SmaFixture"
        config["strategy"]["id"] = scenario.strategy_id
        config["strategy"]["parameters"] = scenario.parameters
        config["backtest"].update(
            symbol="BTC/USDT",
            timeframe="1h",
            start_date="2024-01-01",
            end_date="2024-01-08",
            init_cash=10000,
            fees=0,
            slippage=0,
            mode="signal",
            random_seed=42,
        )
        saved = await call(
            "save_strategy",
            strategy_id=scenario.strategy_id,
            code=code,
            yaml_config=yaml.safe_dump(config),
        )
        assert saved["success"], saved
        for name in ("validate_strategy", "dry_run_strategy"):
            result = await call(name, strategy_id=scenario.strategy_id)
            assert result["success"], result
        started = await call(
            "start_backtest",
            strategy_id=scenario.strategy_id,
            symbol="BTC/USDT",
            timeframe="1h",
            start_date="2024-01-01",
            end_date="2024-01-08",
        )
        assert started["job_id"], started
        finished = await client.wait_for_job(started["job_id"])
        assert finished["status"] == "done", finished
        await call("get_job_status", job_id=started["job_id"])
        run_id = finished["run_id"]
        await call("get_run", run_id=run_id)
        await call("list_artifacts", run_id=run_id)
    return CompletedWorkflow(
        workspace,
        scenario,
        calls,
        Path(saved["py_path"]),
        Path(saved["yaml_path"]),
        run_id,
    )


@pytest.fixture(scope="module", params=["sma", "momentum"])
def completed_workflow(request: pytest.FixtureRequest) -> Iterator[CompletedWorkflow]:
    scenario = SCENARIOS[request.param]
    with temporary_mcp_workspace() as workspace:
        workspace.seed_market(market_frame())
        yield asyncio.run(_complete_workflow(workspace, scenario))
    assert not workspace.root.exists()


def test_workflow_verifier_accepts_completed_backtest(
    completed_workflow: CompletedWorkflow,
) -> None:
    assert_workflow(completed_workflow.calls, completed_workflow.scenario)
    completed_workflow.verify()


def test_workflow_verifier_rejects_incorrect_signals(
    completed_workflow: CompletedWorkflow,
) -> None:
    source = completed_workflow.source_path
    original = source.read_text(encoding="utf-8")
    corrupted = original.replace(
        "        return result\n",
        '        result["signal"] = 0\n        return result\n',
    )
    assert corrupted != original
    try:
        source.write_text(corrupted, encoding="utf-8")
        with pytest.raises(AssertionError, match="Series values are different"):
            completed_workflow.verify()
    finally:
        source.write_text(original, encoding="utf-8")


def test_workflow_verifier_rejects_incorrect_yaml_parameters(
    completed_workflow: CompletedWorkflow,
) -> None:
    config_path = completed_workflow.config_path
    original = config_path.read_text(encoding="utf-8")
    config = yaml.safe_load(original)
    first_parameter = next(iter(config["strategy"]["parameters"]))
    config["strategy"]["parameters"][first_parameter] += 1
    try:
        config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
        with pytest.raises(AssertionError, match="scenario.parameters"):
            completed_workflow.verify()
    finally:
        config_path.write_text(original, encoding="utf-8")


def test_workflow_verifier_rejects_incorrect_stored_metrics(
    completed_workflow: CompletedWorkflow,
) -> None:
    database = completed_workflow.workspace.workspace / "tradingdev.sqlite"
    with closing(sqlite3.connect(database)) as connection:
        row = connection.execute(
            "SELECT metrics FROM runs WHERE run_id = ?", (completed_workflow.run_id,)
        ).fetchone()
        original = row[0]
        metrics = json.loads(original)
        metrics["total_return"] += 1.0
        try:
            with connection:
                connection.execute(
                    "UPDATE runs SET metrics = ? WHERE run_id = ?",
                    (json.dumps(metrics), completed_workflow.run_id),
                )
            with pytest.raises(AssertionError, match="total_return"):
                completed_workflow.verify()
        finally:
            with connection:
                connection.execute(
                    "UPDATE runs SET metrics = ? WHERE run_id = ?",
                    (original, completed_workflow.run_id),
                )


def test_workflow_evidence_requires_model_to_query_result(
    completed_workflow: CompletedWorkflow,
) -> None:
    incomplete = [call for call in completed_workflow.calls if call.name != "get_run"]
    with pytest.raises((AssertionError, StopIteration, ValueError)):
        assert_workflow(incomplete, completed_workflow.scenario)
