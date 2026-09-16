"""Job service tests."""

from __future__ import annotations

import asyncio
import os
from datetime import datetime
from typing import TYPE_CHECKING

import psutil
import pytest

from tradingdev.adapters.execution.process_runner import ProcessIdentity, ProcessRunner
from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.adapters.storage.sqlite import SQLiteStore
from tradingdev.app.data_service import DataService
from tradingdev.app.job_service import JobService
from tradingdev.app.job_store import JobStore

if TYPE_CHECKING:
    from pathlib import Path


class _TerminatesJobService(JobService):
    def _terminate_process(self, identity: ProcessIdentity) -> tuple[bool, str | None]:
        return True, None


class _FakeRunner(ProcessRunner):
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[str, ...]]] = []

    def spawn_module(self, module: str, *args: str) -> ProcessIdentity:
        self.calls.append((module, args))
        return ProcessIdentity(4321, 100.0)


def test_start_walk_forward_uses_bundled_walkforward_config(
    tmp_path: Path,
) -> None:
    workspace = WorkspacePaths(tmp_path / "workspace")
    job_store = JobStore(workspace=workspace, store=SQLiteStore(workspace))
    runner = _FakeRunner()
    service = JobService(
        data_service=DataService(workspace),
        job_store=job_store,
        process_runner=runner,
        project_root=tmp_path,
    )

    response = service.start_walk_forward(
        strategy_id="kd_crossover",
        symbol="BTC/USDT",
        timeframe="1h",
        start_date="2024-01-01",
        end_date="2025-12-31",
    )

    assert response["job_id"]
    job = job_store.get_job(str(response["job_id"]))
    assert job is not None
    assert job["job_type"] == "walk_forward"
    assert job["original_config_path"].endswith(
        "bundled/kd_strategy/walkforward_config.yaml"
    )
    assert runner.calls == [
        (
            "tradingdev.mcp.workers.backtest",
            (
                response["job_id"],
                str(workspace.runs / response["job_id"] / "config.yaml"),
                "--walk-forward",
            ),
        )
    ]


@pytest.mark.parametrize("walk_forward", [False, True])
@pytest.mark.parametrize(
    "failure_type", [OSError, RuntimeError, KeyboardInterrupt, asyncio.CancelledError]
)
def test_start_worker_failure_is_persisted_before_reraising(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    walk_forward: bool,
    failure_type: type[BaseException],
) -> None:
    workspace = WorkspacePaths(tmp_path / "workspace")
    store = JobStore(workspace=workspace)
    runner = _FakeRunner()
    failure = failure_type("worker identity unavailable")

    def fail_spawn(module: str, *args: str) -> ProcessIdentity:
        raise failure

    monkeypatch.setattr(runner, "spawn_module", fail_spawn)
    service = JobService(
        data_service=DataService(workspace),
        job_store=store,
        process_runner=runner,
        project_root=tmp_path,
    )
    start = service.start_walk_forward if walk_forward else service.start_backtest

    with pytest.raises(failure_type) as caught:
        start(
            strategy_id="kd_crossover",
            symbol="BTC/USDT",
            timeframe="1h",
            start_date="2024-01-01",
            end_date="2025-12-31",
        )

    assert caught.value is failure
    jobs = store.list_all_jobs()
    assert len(jobs) == 1
    job = jobs[0]
    assert job["status"] == "failed"
    assert job["job_type"] == ("walk_forward" if walk_forward else "backtest")
    assert job["error"] == (
        f"Worker failed to start: {failure_type.__name__}: worker identity unavailable"
    )
    assert datetime.fromisoformat(job["ended_at"]) >= datetime.fromisoformat(
        job["created_at"]
    )
    assert job["pid"] is None
    assert job["process_create_time"] is None
    response = service.get_job_status(job["job_id"])
    assert response["status"] == "failed"
    assert response["error"] == job["error"]


def test_cancel_job_marks_active_job_cancelled(
    tmp_path: Path,
) -> None:
    workspace = WorkspacePaths(tmp_path / "workspace")
    job_store = JobStore(workspace=workspace, store=SQLiteStore(workspace))
    job_store.create_job(
        job_id="job_cancel",
        strategy_name="fixture",
        symbol="BTC/USDT",
        timeframe="1h",
        start_date="2024-01-01",
        end_date="2024-01-02",
        config_path="fixture.yaml",
    )
    job_store.update_job(
        "job_cancel", status="running_backtest", pid=12345, process_create_time=100.0
    )

    service = _TerminatesJobService(job_store=job_store)

    response = service.cancel_job("job_cancel")

    assert response == {
        "success": True,
        "job_id": "job_cancel",
        "status": "cancelled",
        "process_terminated": True,
    }
    cancelled = job_store.get_job("job_cancel")
    assert cancelled is not None
    assert cancelled["status"] == "cancelled"
    assert cancelled["ended_at"]
    assert cancelled["error"] == "Cancelled by user."


def test_cancel_job_rejects_terminal_job(
    tmp_path: Path,
) -> None:
    workspace = WorkspacePaths(tmp_path / "workspace")
    job_store = JobStore(workspace=workspace, store=SQLiteStore(workspace))
    job_store.create_job(
        job_id="job_done",
        strategy_name="fixture",
        symbol="BTC/USDT",
        timeframe="1h",
        start_date="2024-01-01",
        end_date="2024-01-02",
        config_path="fixture.yaml",
    )
    job_store.update_job("job_done", status="done")

    response = JobService(job_store=job_store).cancel_job("job_done")

    assert response["success"] is False
    assert response["status"] == "done"


def test_get_job_status_returns_run_id_for_completed_optimization(
    tmp_path: Path,
) -> None:
    workspace = WorkspacePaths(tmp_path / "workspace")
    job_store = JobStore(workspace=workspace, store=SQLiteStore(workspace))
    job_store.create_job(
        job_id="job_optimization",
        strategy_name="fixture",
        symbol="BTC/USDT",
        timeframe="1h",
        start_date="2024-01-01",
        end_date="2024-01-02",
        config_path="fixture.yaml",
    )
    job_store.save_result(
        "job_optimization",
        {
            "best_params": {"window": 10},
            "train_metrics": {"sharpe_ratio": 1.0},
            "test_metrics": {"sharpe_ratio": 0.8},
            "optimization_metric": "sharpe_ratio",
            "total_combinations": 3,
        },
    )
    job_store.update_job(
        "job_optimization",
        status="done",
        job_type="optimization",
    )

    response = JobService(job_store=job_store).get_job_status("job_optimization")

    assert response["status"] == "done"
    assert response["job_type"] == "optimization"
    assert response["run_id"] == "job_optimization"
    assert response["best_params"] == {"window": 10}


@pytest.mark.parametrize("created_at", [None, 100.0])
def test_cancel_does_not_signal_reused_or_unidentified_worker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    created_at: float | None,
) -> None:
    workspace = WorkspacePaths(tmp_path / "workspace")
    store = JobStore(workspace=workspace)
    store.create_job(job_id="old_worker")
    store.update_job(
        "old_worker",
        status="running_backtest",
        pid=12345,
        process_create_time=created_at,
    )

    class ReplacementProcess:
        def create_time(self) -> float:
            return 200.0

    monkeypatch.setattr(psutil, "Process", lambda _pid: ReplacementProcess())
    signals: list[tuple[int, int]] = []
    monkeypatch.setattr(os, "kill", lambda pid, sig: signals.append((pid, sig)))
    monkeypatch.setattr(os, "killpg", lambda pid, sig: signals.append((pid, sig)))

    result = JobService(job_store=store).cancel_job("old_worker")

    assert result["success"] is True
    assert result["process_terminated"] is False
    assert not signals


def test_job_status_detects_reused_pid_as_terminated_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = WorkspacePaths(tmp_path / "workspace")
    store = JobStore(workspace=workspace)
    store.create_job(job_id="old_worker")
    store.update_job(
        "old_worker", status="running_backtest", pid=12345, process_create_time=100.0
    )

    class ReplacementProcess:
        def create_time(self) -> float:
            return 200.0

    monkeypatch.setattr(psutil, "Process", lambda _pid: ReplacementProcess())

    result = JobService(job_store=store).get_job_status("old_worker")

    assert result["status"] == "failed"
    assert result["error"] == "Worker process terminated unexpectedly."
    job = store.get_job("old_worker")
    assert job is not None
    assert job["status"] == "failed"
    assert job["error"] == "Worker process terminated unexpectedly."


def test_permission_failure_does_not_claim_worker_was_cancelled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = WorkspacePaths(tmp_path / "workspace")
    store = JobStore(workspace=workspace)
    store.create_job(job_id="worker")
    store.update_job(
        "worker", status="running_backtest", pid=12345, process_create_time=100.0
    )

    def denied(pid: int) -> psutil.Process:
        raise psutil.AccessDenied(pid)

    monkeypatch.setattr(psutil, "Process", denied)
    service = JobService(job_store=store)

    result = service.cancel_job("worker")

    assert result["success"] is False
    assert "Permission denied" in result["error"]
    with pytest.raises(psutil.AccessDenied):
        service.get_job_status("worker")
    job = store.get_job("worker")
    assert job is not None
    assert job["status"] == "running_backtest"


@pytest.mark.parametrize("group_leader", [True, False])
def test_cancel_signals_only_matching_process_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    group_leader: bool,
) -> None:
    workspace = WorkspacePaths(tmp_path / "workspace")
    store = JobStore(workspace=workspace)
    store.create_job(job_id="worker")
    store.update_job(
        "worker", status="running_backtest", pid=12345, process_create_time=100.0
    )
    signals: list[str] = []

    class OriginalProcess:
        def create_time(self) -> float:
            return 100.0

        def is_running(self) -> bool:
            return True

        def status(self) -> str:
            return str(psutil.STATUS_RUNNING)

        def terminate(self) -> None:
            signals.append("process")

    monkeypatch.setattr(psutil, "Process", lambda _pid: OriginalProcess())
    monkeypatch.setattr(os, "getpgid", lambda _pid: 12345 if group_leader else 6789)
    monkeypatch.setattr(os, "killpg", lambda _pid, _sig: signals.append("group"))

    response = JobService(job_store=store).cancel_job("worker")

    assert response["success"] is True
    assert response["process_terminated"] is True
    assert signals == ["group" if group_leader else "process"]
