"""Invalid fixed execution settings are structured rejections, not tool failures."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import pytest
from mcp.server.fastmcp import FastMCP

from tradingdev.adapters.execution.process_runner import ProcessRunner
from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.app.job_service import JobService
from tradingdev.app.job_store import JobStore
from tradingdev.app.optimization_service import OptimizationService
from tradingdev.domain.strategies.loader import StrategyLoader
from tradingdev.mcp.tools import backtest, optimization
from tradingdev.shared.utils.config import load_config

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize(
    "tool", ["start_backtest", "start_walk_forward", "start_optimization"]
)
@pytest.mark.parametrize("invalid", ["nonfinite", "invalid_parallel"])
def test_invalid_execution_config_is_rejected_before_job_creation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tool: str, invalid: str
) -> None:
    store = JobStore(workspace=WorkspacePaths(tmp_path / "workspace"))
    job_service = JobService(job_store=store)
    server = FastMCP("execution-errors")
    backtest.register(server, job_service)
    optimization.register(server, OptimizationService(job_store=store), job_service)

    def invalid_config(path: Path) -> dict[str, Any]:
        config = load_config(path)
        if invalid == "nonfinite":
            config["backtest"]["fees"] = float("nan")
        else:
            config["parallel"] = {"reserve_cores": "not-an-integer"}
        if tool == "start_walk_forward":
            config["validation"] = {}
        return config

    module = "optimization_service" if tool == "start_optimization" else "job_service"
    monkeypatch.setattr(f"tradingdev.app.{module}.load_config", invalid_config)

    def unexpected_spawn(*_args: object) -> None:
        pytest.fail("Rejected execution settings must not start a worker")

    monkeypatch.setattr(ProcessRunner, "spawn_module", unexpected_spawn)
    arguments: dict[str, Any] = {
        "strategy_id": "kd_crossover",
        "symbol": "BTC/USDT",
        "timeframe": "1h",
    }
    if tool == "start_optimization":
        arguments.update(
            param_ranges={"k_period": [3, 5]},
            optimization_metric="total_return",
            train_start="2024-01-01",
            train_end="2024-01-03",
            test_start="2024-01-04",
            test_end="2024-01-07",
        )
    else:
        arguments.update(start_date="2024-01-01", end_date="2024-01-07")
    result = asyncio.run(server.call_tool(tool, arguments))
    assert isinstance(result, tuple)
    payload = result[1]["result"]
    assert payload["job_id"] == ""
    expected = (
        "invalid_optimization_request"
        if tool == "start_optimization"
        else "invalid_execution_request"
    )
    assert payload["code"] == expected
    assert store.list_all_jobs() == []
    assert not list(store.workspace.runs.iterdir())


@pytest.mark.parametrize(
    ("tool", "loader_method"),
    [
        ("start_backtest", "resolve_execution"),
        ("start_walk_forward", "resolve_execution"),
        ("start_optimization", "resolve_execution"),
        ("start_optimization", "validate_parameter_grid"),
    ],
)
@pytest.mark.parametrize("failure_type", [ImportError, RuntimeError, OSError])
def test_strategy_loading_failure_is_structured_before_job_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tool: str,
    loader_method: str,
    failure_type: type[Exception],
) -> None:
    store = JobStore(workspace=WorkspacePaths(tmp_path / "workspace"))
    job_service = JobService(job_store=store)
    server = FastMCP("execution-loading-errors")
    backtest.register(server, job_service)
    optimization.register(server, OptimizationService(job_store=store), job_service)

    def fail_loading(*_args: object) -> None:
        raise failure_type("Strategy module cannot be loaded at submission")

    monkeypatch.setattr(StrategyLoader, loader_method, fail_loading)

    def unexpected_spawn(*_args: object) -> None:
        pytest.fail("Rejected strategy loading must not start a worker")

    monkeypatch.setattr(ProcessRunner, "spawn_module", unexpected_spawn)
    arguments: dict[str, Any] = {
        "strategy_id": "kd_crossover",
        "symbol": "BTC/USDT",
        "timeframe": "1h",
    }
    if tool == "start_optimization":
        arguments.update(
            param_ranges={"k_period": [3, 5]},
            optimization_metric="total_return",
            train_start="2024-01-01",
            train_end="2024-01-03",
            test_start="2024-01-04",
            test_end="2024-01-07",
        )
    else:
        arguments.update(start_date="2024-01-01", end_date="2024-01-07")

    result = asyncio.run(server.call_tool(tool, arguments))
    assert isinstance(result, tuple)
    payload = result[1]["result"]
    assert payload["job_id"] == ""
    assert "Strategy module cannot be loaded at submission" in payload["message"]
    assert payload["code"] == (
        "invalid_optimization_request"
        if tool == "start_optimization"
        else "invalid_execution_request"
    )
    assert store.list_all_jobs() == []
    assert not list(store.workspace.runs.iterdir())
