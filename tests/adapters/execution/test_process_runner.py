"""Process runner adapter tests."""

from __future__ import annotations

import subprocess
import sys
from typing import TYPE_CHECKING

from tradingdev.adapters.execution.process_runner import ProcessRunner
from tradingdev.adapters.storage.filesystem import WorkspacePaths

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


class _FakeProcess:
    pid = 4321


def test_spawn_module_uses_current_interpreter_and_explicit_workspace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[list[str], dict[str, object]]] = []

    def fake_popen(args: list[str], **kwargs: object) -> _FakeProcess:
        calls.append((args, kwargs))
        return _FakeProcess()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)

    workspace = WorkspacePaths(tmp_path / "runtime")
    monkeypatch.setenv("TRADINGDEV_WORKSPACE", str(tmp_path / "unrelated"))
    pid = ProcessRunner(tmp_path, workspace=workspace).spawn_module(
        "tradingdev.mcp.workers.backtest",
        "--job-id",
        "job_123",
    )

    assert pid == 4321
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
