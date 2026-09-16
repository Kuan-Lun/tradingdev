"""Process runner adapter tests."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from typing import TYPE_CHECKING
from unittest.mock import Mock

import psutil
import pytest

from tradingdev.adapters.execution import process_runner
from tradingdev.adapters.execution.process_runner import (
    ProcessIdentity,
    ProcessRunner,
    WorkerHandle,
    request_worker_stop,
)
from tradingdev.adapters.storage.filesystem import WorkspacePaths

if TYPE_CHECKING:
    from pathlib import Path


class _FakeProcess:
    pid = 4321

    def create_time(self) -> float:
        return 100.0


def test_spawn_module_uses_current_interpreter_and_explicit_workspace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[list[str], dict[str, object]]] = []

    def fake_popen(args: list[str], **kwargs: object) -> _FakeProcess:
        calls.append((args, kwargs))
        return _FakeProcess()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    monkeypatch.setattr(psutil, "Process", lambda _pid: _FakeProcess())
    monkeypatch.setattr(
        process_runner, "_wait_for_control", lambda *_: {"worker_pid": 1}
    )

    workspace = WorkspacePaths(tmp_path / "runtime")
    monkeypatch.setenv("TRADINGDEV_WORKSPACE", str(tmp_path / "unrelated"))
    identity = ProcessRunner(tmp_path, workspace=workspace).spawn_module(
        "tradingdev.mcp.workers.backtest",
        "--job-id",
        "job_123",
    )

    assert identity.pid == 4321
    assert identity.create_time == 100.0
    assert calls[0][0] == [
        sys.executable,
        "-m",
        "tradingdev.adapters.execution.worker_supervisor",
        str(workspace.root / ".workers" / identity.control_id),
        "tradingdev.mcp.workers.backtest",
        "--job-id",
        "job_123",
    ]
    assert calls[0][1]["cwd"] == str(tmp_path.resolve())
    assert calls[0][1]["start_new_session"] is True
    env = calls[0][1]["env"]
    assert isinstance(env, dict)
    assert env["TRADINGDEV_WORKSPACE"] == str(workspace.root)
    assert (
        json.loads(
            (
                workspace.root / ".workers" / identity.control_id / "start.json"
            ).read_text()
        )
        == identity.job_fields()
    )


def test_worker_environment_resolves_relative_paths_before_changing_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    def fake_popen(args: list[str], **kwargs: object) -> _FakeProcess:
        calls.append(kwargs)
        return _FakeProcess()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    monkeypatch.setattr(psutil, "Process", lambda _pid: _FakeProcess())
    monkeypatch.setattr(
        process_runner, "_wait_for_control", lambda *_: {"worker_pid": 1}
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TRADINGDEV_WORKSPACE", "runtime")
    monkeypatch.setenv("TRADINGDEV_DATA_ROOT", "market-data")
    monkeypatch.setenv("TRADINGDEV_PROJECT_ROOT", "project")

    runner = ProcessRunner()
    runner.spawn_module("tradingdev.mcp.workers.backtest")

    assert calls[0]["cwd"] == str(tmp_path / "project")
    env = calls[0]["env"]
    assert isinstance(env, dict)
    assert env["TRADINGDEV_WORKSPACE"] == str(tmp_path / "runtime")
    assert env["TRADINGDEV_DATA_ROOT"] == str(tmp_path / "market-data")


@pytest.mark.parametrize(
    ("pid", "created_at"),
    [
        (None, 100.0),
        (0, 100.0),
        (-1, 100.0),
        (True, 100.0),
        (4321, None),
        (4321, "100"),
        (4321, True),
        (4321, 0),
        (4321, -1),
        (4321, float("nan")),
        (4321, float("inf")),
    ],
)
def test_invalid_persisted_process_identity_is_rejected(
    pid: object, created_at: object
) -> None:
    assert ProcessIdentity.from_values(pid, created_at) is None


@pytest.mark.parametrize(
    ("created_at", "running", "status", "matches"),
    [
        (100.0, True, psutil.STATUS_RUNNING, True),
        (200.0, True, psutil.STATUS_RUNNING, False),
        (100.0, False, psutil.STATUS_RUNNING, False),
        (100.0, True, psutil.STATUS_ZOMBIE, False),
    ],
)
def test_process_resolution_checks_persisted_creation_identity(
    monkeypatch: pytest.MonkeyPatch,
    created_at: float,
    running: bool,
    status: str,
    matches: bool,
) -> None:
    class CurrentProcess:
        def create_time(self) -> float:
            return created_at

        def is_running(self) -> bool:
            return running

        def status(self) -> str:
            return status

    process = CurrentProcess()
    monkeypatch.setattr(psutil, "Process", lambda _pid: process)

    resolved = ProcessIdentity(4321, 100.0).get_process()

    assert (resolved is process) is matches
    if not matches:
        assert resolved is None


def test_process_resolution_does_not_hide_permission_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def denied(pid: int) -> psutil.Process:
        raise psutil.AccessDenied(pid)

    monkeypatch.setattr(psutil, "Process", denied)
    with pytest.raises(psutil.AccessDenied):
        ProcessIdentity(4321, 100.0).get_process()


def test_process_resolution_accepts_that_original_worker_exited(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing(pid: int) -> psutil.Process:
        raise psutil.NoSuchProcess(pid)

    monkeypatch.setattr(psutil, "Process", missing)
    assert ProcessIdentity(4321, 100.0).get_process() is None


def test_spawn_cleans_owned_group_before_start_when_identity_capture_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    class Child:
        pid = 4321

    def denied(pid: int) -> psutil.Process:
        raise psutil.AccessDenied(pid)

    def terminate(child: Child) -> int:
        assert child.pid == 4321
        assert not list((tmp_path / ".workers").glob("*/start.json"))
        calls.append("owned-group-cleanup")
        return -9

    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: Child())
    monkeypatch.setattr(psutil, "Process", denied)
    monkeypatch.setattr(process_runner, "terminate_owned_group", terminate)

    with pytest.raises(psutil.AccessDenied):
        ProcessRunner(tmp_path, workspace=WorkspacePaths(tmp_path)).spawn_module(
            "tradingdev.mcp.workers.backtest"
        )

    assert calls == ["owned-group-cleanup"]
    assert not list((tmp_path / ".workers").iterdir())


def test_start_failure_after_handshake_uses_supervisor_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    child = Mock(pid=4321)
    events: list[str] = []
    child.wait.side_effect = lambda **_: events.append("reap-supervisor")
    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: child)
    monkeypatch.setattr(psutil, "Process", lambda _pid: _FakeProcess())
    monkeypatch.setattr(
        process_runner, "_wait_for_control", lambda *_: {"error": "spawn denied"}
    )
    direct_signal = Mock()
    monkeypatch.setattr(process_runner, "terminate_owned_group", direct_signal)

    def stop(root: Path, handle: WorkerHandle, *, timeout: float) -> bool:
        assert (root / ".workers" / handle.control_id / "start.json").exists()
        events.append("worker-cleaned")
        return True

    monkeypatch.setattr(process_runner, "request_worker_stop", stop)
    with pytest.raises(RuntimeError, match="spawn denied"):
        ProcessRunner(tmp_path, workspace=WorkspacePaths(tmp_path)).spawn_module(
            "probe_worker"
        )

    assert events == ["worker-cleaned", "reap-supervisor"]
    direct_signal.assert_not_called()
    assert not list((tmp_path / ".workers").iterdir())


def _control(tmp_path: Path) -> tuple[WorkerHandle, Path]:
    handle = WorkerHandle(4321, 100.0, "a" * 32)
    directory = tmp_path / ".workers" / handle.control_id
    directory.mkdir(parents=True)
    (directory / "start.json").write_text(json.dumps(handle.job_fields()))
    return handle, directory


def test_completed_worker_cleanup_never_targets_a_reused_pid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    handle, directory = _control(tmp_path)
    (directory / "finished.json").write_text('{"cleaned": true}')

    def forbidden(*args: object) -> None:
        pytest.fail("External cleanup must never resolve or signal a numeric PID")

    replacement = Mock()
    replacement.create_time.return_value = 200.0
    monkeypatch.setattr(psutil, "Process", lambda _pid: replacement)
    monkeypatch.setattr(os, "kill", forbidden)
    monkeypatch.setattr(os, "killpg", forbidden)

    assert request_worker_stop(tmp_path, handle) is False
    assert not (directory / "stop").exists()


def test_cancel_waits_for_supervisor_ack_without_pid_signals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    handle, directory = _control(tmp_path)

    def acknowledge(path: Path, timeout: float) -> dict[str, bool]:
        assert path == directory / "finished.json"
        assert (directory / "stop").exists()
        return {"cleaned": True}

    def forbidden(*args: object) -> None:
        pytest.fail("Cancellation must be addressed to a launch token")

    monkeypatch.setattr(ProcessIdentity, "get_process", lambda _: None)
    monkeypatch.setattr(os, "killpg", forbidden)
    monkeypatch.setattr(process_runner, "_wait_for_control", acknowledge)
    assert request_worker_stop(tmp_path, handle) is True


def test_cleanup_acknowledgement_waits_for_supervisor_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    handle, directory = _control(tmp_path)
    (directory / "finished.json").write_text('{"cleaned": true}')
    inspect = Mock(side_effect=[Mock(), None])
    monkeypatch.setattr(ProcessIdentity, "get_process", inspect)
    sleep = Mock()
    monkeypatch.setattr(time, "sleep", sleep)

    assert request_worker_stop(tmp_path, handle) is False
    assert inspect.call_count == 2
    sleep.assert_called_once_with(0.05)


def test_cleanup_acknowledgement_does_not_hide_a_stuck_supervisor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    handle, directory = _control(tmp_path)
    (directory / "finished.json").write_text('{"cleaned": true}')
    monkeypatch.setattr(ProcessIdentity, "get_process", lambda _: Mock())

    with pytest.raises(TimeoutError, match="did not exit after cleanup"):
        request_worker_stop(tmp_path, handle, timeout=0)
    assert directory.exists()


@pytest.mark.parametrize("failure", ["missing", "mismatched", "failed", "timeout"])
def test_unverified_cleanup_fails_and_preserves_control_evidence(
    tmp_path: Path, failure: str
) -> None:
    handle, directory = _control(tmp_path)
    if failure == "missing":
        (directory / "start.json").unlink()
    elif failure == "mismatched":
        (directory / "start.json").write_text('{"pid": 4321}')
    elif failure == "failed":
        (directory / "finished.json").write_text(
            '{"cleaned": false, "error": "group inspection denied"}'
        )
    with pytest.raises((RuntimeError, TimeoutError)):
        request_worker_stop(tmp_path, handle, timeout=0)
    assert directory.exists()
    if failure in {"missing", "mismatched"}:
        assert not (directory / "stop").exists()
