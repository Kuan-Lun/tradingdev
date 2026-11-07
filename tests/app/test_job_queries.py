"""Job discovery reads saved state; only an explicit poll recovers workers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.app.job_service import JobService
from tradingdev.app.job_store import JobStore

if TYPE_CHECKING:
    from pathlib import Path


def test_inventory_does_not_reconcile_a_missing_worker(tmp_path: Path) -> None:
    store = JobStore(workspace=WorkspacePaths(tmp_path))
    store.create_job(job_id="missing_worker")
    store.update_job("missing_worker", status="running_backtest")
    before = store.get_job("missing_worker")
    service = JobService(job_store=store)

    listed = service.list_jobs()
    page = service.list_jobs_page(status="running_backtest")

    assert len(listed) == page.total == 1
    assert listed[0]["status"] == page.items[0].status == "running_backtest"
    assert store.get_job("missing_worker") == before

    polled = service.get_job_status("missing_worker")
    assert polled["status"] == "failed"
    assert polled["error"] == "Worker process terminated unexpectedly."
    after = store.get_job("missing_worker")
    assert after is not None and after["status"] == "failed" and after["ended_at"]
    assert service.list_jobs_page(status="running_backtest").total == 0
