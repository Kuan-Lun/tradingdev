"""Process runner adapter tests."""

from __future__ import annotations

import subprocess
import sys
from typing import TYPE_CHECKING

import psutil
import pytest

from tradingdev.adapters.execution.process_runner import ProcessIdentity, ProcessRunner
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

    workspace = WorkspacePaths(tmp_path / "runtime")
    monkeypatch.setenv("TRADINGDEV_WORKSPACE", str(tmp_path / "unrelated"))
    identity = ProcessRunner(tmp_path, workspace=workspace).spawn_module(
        "tradingdev.mcp.workers.backtest",
        "--job-id",
        "job_123",
    )

    assert identity == ProcessIdentity(4321, 100.0)
    assert calls[0][0] == [
        sys.executable,
        "-m",
        "tradingdev.mcp.workers.backtest",
        "--job-id",
        "job_123",
    ]
    assert calls[0][1]["cwd"] == str(tmp_path.resolve())
    assert calls[0][1]["start_new_session"] is True
    env = calls[0][1]["env"]
    assert isinstance(env, dict)
    assert env["TRADINGDEV_WORKSPACE"] == str(workspace.root)


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


def test_spawn_reaps_child_when_process_identity_capture_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    class Child:
        pid = 4321

        def kill(self) -> None:
            calls.append("kill")

        def wait(self, *, timeout: int) -> None:
            assert timeout == 5
            calls.append("wait")

    def denied(pid: int) -> psutil.Process:
        raise psutil.AccessDenied(pid)

    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: Child())
    monkeypatch.setattr(psutil, "Process", denied)

    with pytest.raises(psutil.AccessDenied):
        ProcessRunner(tmp_path).spawn_module("tradingdev.mcp.workers.backtest")

    assert calls == ["kill", "wait"]
