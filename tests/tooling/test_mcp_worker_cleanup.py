"""Deterministic PID reuse regressions for the MCP test harness."""

from __future__ import annotations

import os
import shutil
import signal
from typing import TYPE_CHECKING
from unittest.mock import Mock

import psutil
import pytest
from tests.integration import mcp_harness

from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.app.job_store import JobStore

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def worker_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[mcp_harness.MCPWorkspace, JobStore, Mock, Mock]:
    workspace = mcp_harness.MCPWorkspace(tmp_path)
    store = JobStore(workspace=WorkspacePaths(workspace.workspace))
    store.create_job(job_id="original")
    store.update_job("original", pid=4321, process_create_time=100.0, status="done")
    process = Mock(spec=psutil.Process)
    process.pid = 4321
    process.create_time.return_value = 100.0
    process.cmdline.return_value = [
        "python",
        "-m",
        "tradingdev.mcp.workers.optimization",
        "original",
    ]
    process.cwd.return_value = str(workspace.root)
    process.is_running.return_value = True
    process.status.return_value = psutil.STATUS_SLEEPING
    monkeypatch.setattr(psutil, "Process", Mock(return_value=process))
    monkeypatch.setattr(os, "getpgid", Mock(return_value=4321))

    def stop_group(pid: int, sig: signal.Signals) -> None:
        process.status.return_value = psutil.STATUS_ZOMBIE

    killpg = Mock(side_effect=stop_group)
    monkeypatch.setattr(os, "killpg", killpg)
    return workspace, store, process, killpg


@pytest.mark.parametrize("status", ["done", "running_backtest"])
def test_cleanup_does_not_signal_reused_pid_in_same_workspace(
    worker_cleanup: tuple[mcp_harness.MCPWorkspace, JobStore, Mock, Mock],
    status: str,
) -> None:
    workspace, store, process, killpg = worker_cleanup
    store.update_job("original", status=status)
    process.create_time.return_value = 200.0
    process.cmdline.return_value[-1] = "replacement"

    workspace.stop_workers()

    killpg.assert_not_called()


@pytest.mark.parametrize("status", ["done", "pending_confirmation"])
def test_cleanup_stops_matching_worker_even_when_job_is_terminal(
    worker_cleanup: tuple[mcp_harness.MCPWorkspace, JobStore, Mock, Mock],
    status: str,
) -> None:
    workspace, store, _, killpg = worker_cleanup
    store.update_job("original", status=status)

    workspace.stop_workers()

    killpg.assert_called_once_with(4321, signal.SIGTERM)


def test_cleanup_reports_missing_identity_without_signalling(
    worker_cleanup: tuple[mcp_harness.MCPWorkspace, JobStore, Mock, Mock],
) -> None:
    workspace, store, _, killpg = worker_cleanup
    store.update_job("original", process_create_time=None)

    with pytest.raises(AssertionError, match="Worker creation identity missing"):
        workspace.stop_workers()

    killpg.assert_not_called()


def test_cleanup_rechecks_identity_before_sending_signal(
    worker_cleanup: tuple[mcp_harness.MCPWorkspace, JobStore, Mock, Mock],
) -> None:
    workspace, _, process, killpg = worker_cleanup
    # psutil's retained Process detects replacement after it was inspected.
    process.is_running.side_effect = [True, False, False]

    workspace.stop_workers()

    killpg.assert_not_called()


def test_cleanup_reports_identity_inspection_failure_without_signalling(
    worker_cleanup: tuple[mcp_harness.MCPWorkspace, JobStore, Mock, Mock],
) -> None:
    workspace, _, process, killpg = worker_cleanup
    process.create_time.side_effect = psutil.AccessDenied(process.pid)

    with pytest.raises(psutil.AccessDenied):
        workspace.stop_workers()

    killpg.assert_not_called()


def test_cleanup_stops_known_workers_before_reporting_an_unidentified_worker(
    worker_cleanup: tuple[mcp_harness.MCPWorkspace, JobStore, Mock, Mock],
) -> None:
    workspace, store, _, killpg = worker_cleanup
    store.update_job("original", process_create_time=None)
    store.create_job(job_id="known")
    store.update_job("known", pid=4321, process_create_time=100.0)

    with pytest.raises(AssertionError, match="Worker creation identity missing"):
        workspace.stop_workers()

    killpg.assert_called_once_with(4321, signal.SIGTERM)


def test_workspace_retains_files_when_worker_cleanup_cannot_be_verified(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_cleanup(self: mcp_harness.MCPWorkspace) -> None:
        raise AssertionError("Worker creation identity missing")

    monkeypatch.setattr(mcp_harness.MCPWorkspace, "stop_workers", fail_cleanup)
    directory: Path | None = None
    try:
        with (
            pytest.raises(RuntimeError, match="temporary directory retained"),
            mcp_harness.temporary_mcp_workspace() as workspace,
        ):
            directory = workspace.root
            (directory / "worker-data").write_text("in use", encoding="utf-8")
        assert directory is not None
        assert (directory / "worker-data").read_text(encoding="utf-8") == "in use"
    finally:
        # No process was started; remove this intentionally retained fixture.
        if directory is not None:
            shutil.rmtree(directory)
            assert not directory.exists()
