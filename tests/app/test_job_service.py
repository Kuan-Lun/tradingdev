"""Job service tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.adapters.storage.sqlite import SQLiteStore
from tradingdev.app.job_service import JobService
from tradingdev.app.job_store import JobStore

if TYPE_CHECKING:
    from pathlib import Path


class _TerminatesJobService(JobService):
    def _is_pid_alive(self, pid: object) -> bool:
        return True

    def _terminate_pid(self, pid: int) -> tuple[bool, str | None]:
        return True, None


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
    job_store.update_job("job_cancel", status="running_backtest", pid=12345)

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
