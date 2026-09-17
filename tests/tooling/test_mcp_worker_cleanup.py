"""Regressions for workspace-owned worker cleanup without PID signalling."""

from __future__ import annotations

import json
import os
import shutil
from typing import TYPE_CHECKING
from unittest.mock import Mock, call

import psutil
import pytest
from tests.integration import mcp_harness

from tradingdev.adapters.execution.process_runner import (
    WorkerHandle,
    request_worker_stop,
)
from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.app.job_store import JobStore

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


@pytest.fixture(autouse=True)
def forbid_pid_signals(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    kill = Mock()
    killpg = Mock()
    monkeypatch.setattr(os, "kill", kill)
    monkeypatch.setattr(os, "killpg", killpg, raising=False)
    yield
    kill.assert_not_called()
    killpg.assert_not_called()


@pytest.fixture
def worker_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[mcp_harness.MCPWorkspace, JobStore, WorkerHandle, Mock]:
    workspace = mcp_harness.MCPWorkspace(tmp_path)
    store = JobStore(workspace=WorkspacePaths(workspace.workspace))
    handle = WorkerHandle(pid=4321, create_time=100.0, control_id="a" * 32)
    store.create_job(job_id="original")
    store.update_job("original", **handle.job_fields(), status="done")
    stop = Mock(return_value=True)
    monkeypatch.setattr(mcp_harness, "request_worker_stop", stop)
    return workspace, store, handle, stop


def _record_control(workspace: mcp_harness.MCPWorkspace, handle: WorkerHandle) -> Path:
    start = workspace.workspace / ".workers" / handle.control_id / "start.json"
    start.parent.mkdir(parents=True, exist_ok=True)
    start.write_text(json.dumps(handle.job_fields()), encoding="utf-8")
    return start


@pytest.mark.parametrize("status", ["done", "failed", "pending_confirmation"])
def test_cleanup_requests_matching_worker_even_when_job_is_terminal(
    worker_cleanup: tuple[mcp_harness.MCPWorkspace, JobStore, WorkerHandle, Mock],
    status: str,
) -> None:
    workspace, store, handle, stop = worker_cleanup
    store.update_job("original", status=status)

    workspace.stop_workers()

    stop.assert_called_once_with(workspace.workspace, handle)


@pytest.mark.parametrize("status", ["done", "running_backtest"])
def test_cleanup_does_not_signal_reused_pid_after_original_completed(
    worker_cleanup: tuple[mcp_harness.MCPWorkspace, JobStore, WorkerHandle, Mock],
    monkeypatch: pytest.MonkeyPatch,
    status: str,
) -> None:
    workspace, store, handle, _ = worker_cleanup
    store.update_job("original", status=status)
    replacement = Mock(spec=psutil.Process)
    replacement.create_time.return_value = 200.0
    process = Mock(return_value=replacement)
    monkeypatch.setattr(psutil, "Process", process)
    # The original supervisor has already acknowledged complete cleanup.
    start = _record_control(workspace, handle)
    start.with_name("finished.json").write_text('{"cleaned": true}', encoding="utf-8")
    monkeypatch.setattr(mcp_harness, "request_worker_stop", request_worker_stop)

    workspace.stop_workers()

    process.assert_called_once_with(handle.pid)
    replacement.create_time.assert_called_once_with()
    assert not start.with_name("stop").exists()


def test_cleanup_deduplicates_job_and_control_records(
    worker_cleanup: tuple[mcp_harness.MCPWorkspace, JobStore, WorkerHandle, Mock],
) -> None:
    workspace, store, handle, stop = worker_cleanup
    _record_control(workspace, handle)
    store.create_job(job_id="duplicate")
    store.update_job("duplicate", **handle.job_fields())

    workspace.stop_workers()

    stop.assert_called_once_with(workspace.workspace, handle)


def test_cleanup_finds_worker_started_before_database_update(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = mcp_harness.MCPWorkspace(tmp_path)
    handle = WorkerHandle(pid=4321, create_time=100.0, control_id="b" * 32)
    _record_control(workspace, handle)
    stop = Mock(return_value=True)
    monkeypatch.setattr(mcp_harness, "request_worker_stop", stop)
    assert not (workspace.workspace / "tradingdev.sqlite").exists()

    workspace.stop_workers()

    stop.assert_called_once_with(workspace.workspace, handle)


@pytest.mark.parametrize("finished", [False, True])
def test_cleanup_reports_launch_without_identity_but_stops_known_workers(
    worker_cleanup: tuple[mcp_harness.MCPWorkspace, JobStore, WorkerHandle, Mock],
    finished: bool,
) -> None:
    workspace, _, handle, stop = worker_cleanup
    unidentified = workspace.workspace / ".workers" / ("b" * 32)
    unidentified.mkdir(parents=True)
    if finished:
        (unidentified / "finished.json").write_text(
            '{"cleaned": true}', encoding="utf-8"
        )

    with pytest.raises(AssertionError, match="Unidentified worker launch"):
        workspace.stop_workers()

    stop.assert_called_once_with(workspace.workspace, handle)
    assert unidentified.exists()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("process_create_time", None),
        ("worker_control_id", None),
        ("worker_control_id", "../another-workspace"),
    ],
)
def test_cleanup_reports_incomplete_identity_but_stops_other_known_workers(
    worker_cleanup: tuple[mcp_harness.MCPWorkspace, JobStore, WorkerHandle, Mock],
    field: str,
    value: str | None,
) -> None:
    workspace, store, handle, stop = worker_cleanup
    store.update_job("original", **{field: value})
    store.create_job(job_id="known")
    store.update_job("known", **handle.job_fields())

    with pytest.raises(
        AssertionError, match="Worker control identity missing or invalid"
    ):
        workspace.stop_workers()

    stop.assert_called_once_with(workspace.workspace, handle)


@pytest.mark.parametrize("contents", ['{"pid": 4321}', "not-json", "[]"])
def test_cleanup_reports_invalid_control_and_stops_other_known_workers(
    worker_cleanup: tuple[mcp_harness.MCPWorkspace, JobStore, WorkerHandle, Mock],
    contents: str,
) -> None:
    workspace, _, handle, stop = worker_cleanup
    invalid = workspace.workspace / ".workers" / ("b" * 32) / "start.json"
    invalid.parent.mkdir(parents=True)
    invalid.write_text(contents, encoding="utf-8")

    with pytest.raises((AssertionError, json.JSONDecodeError)):
        workspace.stop_workers()

    stop.assert_called_once_with(workspace.workspace, handle)


def test_cleanup_rejects_control_record_from_another_directory(
    worker_cleanup: tuple[mcp_harness.MCPWorkspace, JobStore, WorkerHandle, Mock],
) -> None:
    workspace, _, handle, stop = worker_cleanup
    start = _record_control(workspace, handle)
    start.parent.rename(start.parent.with_name("b" * 32))

    with pytest.raises(AssertionError, match="Invalid worker control record"):
        workspace.stop_workers()

    stop.assert_called_once_with(workspace.workspace, handle)


@pytest.mark.parametrize(
    "error",
    [TimeoutError("cleanup timed out"), RuntimeError("supervisor cleanup failed")],
)
def test_cleanup_requests_other_workers_after_control_failure(
    worker_cleanup: tuple[mcp_harness.MCPWorkspace, JobStore, WorkerHandle, Mock],
    error: Exception,
) -> None:
    workspace, store, handle, stop = worker_cleanup
    other = WorkerHandle(pid=5432, create_time=150.0, control_id="b" * 32)
    store.create_job(job_id="other")
    store.update_job("other", **other.job_fields())

    def request(root: Path, worker: WorkerHandle) -> bool:
        if worker == handle:
            raise error
        return True

    stop.side_effect = request

    with pytest.raises(type(error), match=str(error)):
        workspace.stop_workers()

    assert stop.call_count == 2
    stop.assert_has_calls(
        [call(workspace.workspace, handle), call(workspace.workspace, other)],
        any_order=True,
    )


@pytest.mark.parametrize(
    "error",
    [TimeoutError("cleanup timed out"), RuntimeError("supervisor cleanup failed")],
)
def test_workspace_retains_files_when_worker_cleanup_cannot_be_verified(
    monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    monkeypatch.setattr(mcp_harness, "request_worker_stop", Mock(side_effect=error))
    directory: Path | None = None
    try:
        with (
            pytest.raises(RuntimeError, match="temporary directory retained"),
            mcp_harness.temporary_mcp_workspace() as workspace,
        ):
            directory = workspace.root
            handle = WorkerHandle(pid=4321, create_time=100.0, control_id="a" * 32)
            _record_control(workspace, handle)
            (directory / "worker-data").write_text("in use", encoding="utf-8")
        assert directory is not None
        assert (directory / "worker-data").read_text(encoding="utf-8") == "in use"
    finally:
        # No process was started; remove this intentionally retained fixture.
        if directory is not None:
            shutil.rmtree(directory)
            assert not directory.exists()


def test_workspace_retains_files_when_a_launch_has_no_start_record(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stop = Mock(return_value=True)
    monkeypatch.setattr(mcp_harness, "request_worker_stop", stop)
    directory: Path | None = None
    try:
        with (
            pytest.raises(RuntimeError, match="temporary directory retained"),
            mcp_harness.temporary_mcp_workspace() as workspace,
        ):
            directory = workspace.root
            (workspace.workspace / ".workers" / ("b" * 32)).mkdir(parents=True)
            (directory / "worker-data").write_text("in use", encoding="utf-8")
        assert directory is not None
        assert (directory / "worker-data").read_text(encoding="utf-8") == "in use"
        stop.assert_not_called()
    finally:
        # No process was started; remove this intentionally retained fixture.
        if directory is not None and directory.exists():
            shutil.rmtree(directory)
            assert not directory.exists()
