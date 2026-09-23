"""Job contract regressions at the FastMCP boundary."""

from __future__ import annotations

import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any

import pytest
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError

from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.app.job_service import JobService
from tradingdev.app.job_store import JobStore
from tradingdev.mcp.tools import jobs

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture
def job_service(tmp_path: Path) -> Iterator[JobService]:
    with TemporaryDirectory(dir=tmp_path) as directory:
        workspace = WorkspacePaths(Path(directory))
        store = JobStore(workspace=workspace)
        yield JobService(job_store=store)


@pytest.mark.parametrize("corruption", ["extra_field", "unknown_state", "missing_code"])
def test_job_output_rejects_unknown_fields_and_invalid_state(
    job_service: JobService, monkeypatch: pytest.MonkeyPatch, corruption: str
) -> None:
    server = FastMCP("contract-test")
    jobs.register(server, job_service)
    # A service regression must fail at the boundary, not silently publish an
    # arbitrary object or discard an internal field the client should not see.
    malformed = {
        "status": "not_found",
        "error": "missing",
        "code": "job_not_found",
    }
    if corruption == "extra_field":
        malformed["worker_control_id"] = "private"
    elif corruption == "unknown_state":
        malformed["status"] = "unexpected"
    else:
        del malformed["code"]
    monkeypatch.setattr(job_service, "get_job_status", lambda _: malformed)
    with pytest.raises(ToolError):
        asyncio.run(server.call_tool("get_job_status", {"job_id": "missing"}))


@pytest.mark.parametrize(
    "state",
    [
        "queued",
        "downloading_data",
        "running_backtest",
        "estimating",
        "pending_confirmation",
        "optimizing",
        "testing_oos",
        "done",
        "failed",
        "cancelled",
        "estimation_timeout",
    ],
)
def test_job_lifecycle_payloads_satisfy_advertised_contract(
    job_service: JobService, monkeypatch: pytest.MonkeyPatch, state: str
) -> None:
    # Exercise the real service formatting for every worker state without
    # starting a worker. Process liveness is the only substituted dependency.
    store = job_service._job_store
    store.create_job(
        job_id="job",
        strategy_name="fixture",
        symbol="BTC/USDT",
        timeframe="1h",
        start_date="2024-01-01",
        end_date="2024-01-02",
        config_path="missing.yaml",
    )
    store.update_job(
        "job",
        status=state,
        job_type="optimization",
        total_combinations=2,
        completed=1,
        time_per_combo=0.5,
        estimated_total_seconds=1.0,
        n_parallel_workers=1,
        estimated_remaining_seconds=0.5,
        error="fixture failure" if state == "failed" else None,
    )
    if state == "done":
        result_path = store.save_result(
            "job",
            {
                "best_params": {"period": 3},
                "train_metrics": {"sharpe_ratio": 0.0},
                "test_metrics": {"sharpe_ratio": None},
                "optimization_metric": "sharpe_ratio",
                "total_combinations": 2,
            },
        )
        store.update_job("job", result_path=str(result_path))
    monkeypatch.setattr(job_service, "_is_process_alive", lambda _: True)
    server = FastMCP("contract-test")
    jobs.register(server, job_service)
    result = asyncio.run(server.call_tool("get_job_status", {"job_id": "job"}))
    assert isinstance(result, tuple)
    payload: Any = result[1]["result"]
    assert payload["status"] == state
    if state == "done":
        assert payload["train_metrics"]["sharpe_ratio"] == 0.0
        assert payload["test_metrics"]["sharpe_ratio"] is None


def test_cancel_contract_does_not_claim_success_when_identity_missing(
    job_service: JobService,
) -> None:
    store = job_service._job_store
    store.create_job(
        job_id="job",
        strategy_name="fixture",
        symbol="BTC/USDT",
        timeframe="1h",
        start_date="2024-01-01",
        end_date="2024-01-02",
        config_path="missing.yaml",
    )
    store.update_job("job", status="running_backtest", pid=12345)
    server = FastMCP("contract-test")
    jobs.register(server, job_service)
    result = asyncio.run(server.call_tool("cancel_job", {"job_id": "job"}))
    assert isinstance(result, tuple)
    payload = result[1]["result"]
    assert not payload["success"]
    assert payload["code"] == "worker_identity_unavailable"
    assert "process_terminated" not in payload
    persisted = store.get_job("job")
    assert persisted is not None and persisted["status"] == "running_backtest"
