"""Workers preserve their launch supervisor identity in persisted job records."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from tradingdev.adapters.execution.process_runner import WorkerHandle
from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.app.job_store import JobStore
from tradingdev.mcp.workers import backtest, optimization

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize("job_type", ["backtest", "optimization"])
def test_worker_keeps_supervisor_identity_when_starting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, job_type: str
) -> None:
    workspace = WorkspacePaths(tmp_path / "workspace")
    store = JobStore(workspace=workspace)
    handle = WorkerHandle(4321, 100.0, "a" * 32)
    config_path = tmp_path / "missing.yaml"
    store.create_job(
        job_id="worker",
        config_path=str(config_path),
        job_type=job_type,
        extra_payload={
            **handle.job_fields(),
            "param_ranges": {"window": [10]},
            "optimization_metric": "sharpe_ratio",
            "train_start": "2024-01-01",
            "train_end": "2024-02-01",
            "test_start": "2024-02-02",
            "test_end": "2024-03-01",
        },
    )
    monkeypatch.setenv("TRADINGDEV_WORKSPACE", str(workspace.root))
    monkeypatch.setenv("TRADINGDEV_WORKER_IDENTITY", json.dumps(handle.job_fields()))

    # A missing config stops execution after the worker's startup write. This
    # exercises real persisted metadata without running a backtest or child.
    if job_type == "backtest":
        backtest._run_backtest("worker", config_path)
    else:
        optimization._run_optimization("worker")

    job = store.get_job("worker")
    assert job is not None
    assert job["status"] == "failed"
    assert job["started_at"] is not None
    assert WorkerHandle.from_job(job) == handle
