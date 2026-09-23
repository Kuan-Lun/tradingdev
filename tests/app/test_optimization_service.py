"""Optimization service tests."""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import TYPE_CHECKING, Any, cast

import pytest

from tradingdev.adapters.execution.process_runner import WorkerHandle
from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.adapters.storage.sqlite import SQLiteStore
from tradingdev.app.job_service import JobService
from tradingdev.app.job_store import JobStore
from tradingdev.app.optimization_service import OptimizationService
from tradingdev.app.strategy_service import StrategyNotExecutableError
from tradingdev.domain.strategies.schemas import StrategySpec, StrategyStatus

if TYPE_CHECKING:
    from pathlib import Path

    from tradingdev.adapters.execution.process_runner import ProcessRunner
    from tradingdev.app.strategy_service import StrategyService

_EXECUTABLE = {StrategyStatus.RUNNABLE, StrategyStatus.PROMOTED}


class _StrategyServiceStub:
    def __init__(self, metadata: dict[str, Any]) -> None:
        self.metadata = metadata

    def resolve_executable(self, strategy_id: str) -> StrategySpec:
        status = StrategyStatus(str(self.metadata["status"]))
        if status not in _EXECUTABLE:
            msg = (
                "Strategy must be runnable or promoted before execution. "
                f"Current status: {status.value}"
            )
            raise StrategyNotExecutableError(msg)
        return StrategySpec(
            strategy_id=strategy_id,
            class_name="Fixture",
            source_path="",
            config_path=str(self.metadata["config_path"]),
            status=status,
        )


class _RunnerStub:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[str, ...]]] = []

    def spawn_module(self, module: str, *args: str) -> WorkerHandle:
        self.calls.append((module, args))
        return WorkerHandle(2468, 100.0, "a" * 32)


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
    config_path.write_text(
        "strategy:\n  id: fixture\nbacktest:\n  symbol: ETH/USDT\n"
        "  timeframe: 4h\n  start_date: '2024-01-01'\n"
        "  end_date: '2024-12-31'\n  init_cash: 10000\n",
        encoding="utf-8",
    )
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
        test_start="2024-02-02",
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
    assert job["process_create_time"] == 100.0
    assert job["worker_control_id"] == "a" * 32
    assert job["total_combinations"] == 6
    assert job["optimization_metric"] == "sharpe_ratio"
    assert job["param_ranges"] == {
        "window": [10, 20],
        "threshold": [0.1, 0.2, 0.3],
    }


@pytest.mark.parametrize(
    "failure_type", [OSError, RuntimeError, KeyboardInterrupt, asyncio.CancelledError]
)
def test_optimization_spawn_failure_is_persisted_before_reraising(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_type: type[BaseException],
) -> None:
    config_path = tmp_path / "strategy.yaml"
    config_path.write_text(
        "strategy:\n  id: fixture\nbacktest:\n  symbol: ETH/USDT\n"
        "  timeframe: 4h\n  start_date: '2024-01-01'\n"
        "  end_date: '2024-12-31'\n  init_cash: 10000\n",
        encoding="utf-8",
    )
    service, store, runner = _service(
        tmp_path,
        metadata={"status": "runnable", "config_path": str(config_path)},
    )
    failure = failure_type("worker identity unavailable")

    def fail_spawn(module: str, *args: str) -> WorkerHandle:
        raise failure

    monkeypatch.setattr(runner, "spawn_module", fail_spawn)

    with pytest.raises(failure_type) as caught:
        service.start_optimization(
            strategy_id="fixture",
            symbol="BTC/USDT",
            timeframe="1h",
            param_ranges={"window": [10, 20]},
            optimization_metric="sharpe_ratio",
            train_start="2024-01-01",
            train_end="2024-02-01",
            test_start="2024-02-02",
            test_end="2024-03-01",
        )

    assert caught.value is failure
    jobs = store.list_all_jobs()
    assert len(jobs) == 1
    job = jobs[0]
    assert job["status"] == "failed"
    assert job["job_type"] == "optimization"
    assert job["error"] == (
        f"Worker failed to start: {failure_type.__name__}: worker identity unavailable"
    )
    assert datetime.fromisoformat(job["ended_at"]) >= datetime.fromisoformat(
        job["created_at"]
    )
    assert job["pid"] is None
    assert job["process_create_time"] is None
    response = JobService(job_store=store).get_job_status(job["job_id"])
    assert response["status"] == "failed"
    assert response["error"] == job["error"]


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
        "code": "invalid_optimization_request",
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
    assert "must be runnable or promoted" in response["message"]
    assert job_store.list_all_jobs() == []
    assert runner.calls == []
