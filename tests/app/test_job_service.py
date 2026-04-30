"""Job service tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.adapters.storage.sqlite import SQLiteStore
from tradingdev.app import job_store
from tradingdev.app.job_service import JobService

if TYPE_CHECKING:
    from pathlib import Path

    from pytest import MonkeyPatch


def test_cancel_job_marks_active_job_cancelled(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    workspace = WorkspacePaths(tmp_path / "workspace")
    store = SQLiteStore(workspace)
    monkeypatch.setattr(job_store, "_WORKSPACE", workspace)
    monkeypatch.setattr(job_store, "_STORE", store)
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

    service = JobService()
    monkeypatch.setattr(service, "_is_pid_alive", lambda _pid: True)
    monkeypatch.setattr(service, "_terminate_pid", lambda _pid: (True, None))

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
    monkeypatch: MonkeyPatch,
) -> None:
    workspace = WorkspacePaths(tmp_path / "workspace")
    store = SQLiteStore(workspace)
    monkeypatch.setattr(job_store, "_WORKSPACE", workspace)
    monkeypatch.setattr(job_store, "_STORE", store)
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

    response = JobService().cancel_job("job_done")

    assert response["success"] is False
    assert response["status"] == "done"
