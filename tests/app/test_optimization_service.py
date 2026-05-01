"""Optimization service tests."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.adapters.storage.sqlite import SQLiteStore
from tradingdev.app.job_store import JobStore
from tradingdev.app.optimization_service import OptimizationService

if TYPE_CHECKING:
    from pathlib import Path

    from tradingdev.adapters.execution.process_runner import ProcessRunner
    from tradingdev.app.strategy_service import StrategyService


class _StrategyServiceStub:
    def __init__(self, metadata: dict[str, Any]) -> None:
        self.metadata = metadata

    def get_strategy(self, strategy_id: str) -> dict[str, Any]:
        return {"success": True, "metadata": self.metadata, "id": strategy_id}


class _RunnerStub:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[str, ...]]] = []

    def spawn_module(self, module: str, *args: str) -> int:
        self.calls.append((module, args))
        return 2468


def _service(
    tmp_path: Path,
    *,
    metadata: dict[str, Any],
) -> tuple[OptimizationService, JobStore, _RunnerStub]:
    workspace = WorkspacePaths(tmp_path / "workspace")
    job_store = JobStore(workspace=workspace, store=SQLiteStore(workspace))
    runner = _RunnerStub()
    service = OptimizationService(
        strategy_service=cast("StrategyService", _StrategyServiceStub(metadata)),
        job_store=job_store,
        process_runner=cast("ProcessRunner", runner),
        project_root=tmp_path,
    )
    return service, job_store, runner


def test_start_optimization_creates_job_and_spawns_worker(tmp_path: Path) -> None:
    config_path = tmp_path / "strategy.yaml"
    config_path.write_text("strategy:\n  id: fixture\n", encoding="utf-8")
    service, job_store, runner = _service(
        tmp_path,
        metadata={"status": "runnable", "config_path": str(config_path)},
    )

    response = service.start_optimization(
        strategy_id="fixture",
        symbol="BTC/USDT",
        timeframe="1h",
        param_ranges={"window": [10, 20], "threshold": [0.1, 0.2, 0.3]},
        optimization_metric="sharpe_ratio",
        train_start="2024-01-01",
        train_end="2024-02-01",
        test_start="2024-02-01",
        test_end="2024-03-01",
    )

    assert response["job_id"]
    assert response["total_combinations"] == 6
    assert runner.calls == [
        ("tradingdev.mcp.workers.optimization", (response["job_id"],))
    ]
    job = job_store.get_job(str(response["job_id"]))
    assert job is not None
    assert job["job_type"] == "optimization"
    assert job["pid"] == 2468
    assert job["total_combinations"] == 6
    assert job["optimization_metric"] == "sharpe_ratio"
    assert job["param_ranges"] == {
        "window": [10, 20],
        "threshold": [0.1, 0.2, 0.3],
    }


def test_start_optimization_rejects_invalid_request_before_spawning(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "strategy.yaml"
    config_path.write_text("strategy:\n  id: fixture\n", encoding="utf-8")
    service, job_store, runner = _service(
        tmp_path,
        metadata={"status": "runnable", "config_path": str(config_path)},
    )

    response = service.start_optimization(
        strategy_id="fixture",
        symbol="BTC/USDT",
        timeframe="1h",
        param_ranges={"window": []},
        optimization_metric="sortino_ratio",
        train_start="2024-01-01",
        train_end="2024-02-01",
        test_start="2024-02-01",
        test_end="2024-03-01",
    )

    assert response == {
        "job_id": "",
        "message": "param_ranges['window'] must be a non-empty list.",
        "total_combinations": 0,
    }
    assert job_store.list_all_jobs() == []
    assert runner.calls == []


def test_start_optimization_requires_runnable_strategy(tmp_path: Path) -> None:
    service, job_store, runner = _service(
        tmp_path,
        metadata={"status": "draft", "config_path": str(tmp_path / "missing.yaml")},
    )

    response = service.start_optimization(
        strategy_id="fixture",
        symbol="BTC/USDT",
        timeframe="1h",
        param_ranges={"window": [10]},
        optimization_metric="sharpe_ratio",
        train_start="2024-01-01",
        train_end="2024-02-01",
        test_start="2024-02-01",
        test_end="2024-03-01",
    )

    assert response["job_id"] == ""
    assert (
        response["message"]
        == "Strategy must be runnable or promoted before optimization."
    )
    assert job_store.list_all_jobs() == []
    assert runner.calls == []
