"""Process runner adapter tests."""

from __future__ import annotations

import subprocess
from typing import TYPE_CHECKING

from tradingdev.adapters.execution.process_runner import ProcessRunner

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


class _FakeProcess:
    pid = 4321


def test_spawn_module_uses_uv_run_python(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[list[str], dict[str, object]]] = []

    def fake_popen(args: list[str], **kwargs: object) -> _FakeProcess:
        calls.append((args, kwargs))
        return _FakeProcess()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)

    pid = ProcessRunner(tmp_path).spawn_module(
        "tradingdev.mcp.workers.backtest",
        "--job-id",
        "job_123",
    )

    assert pid == 4321
    assert calls[0][0] == [
        "uv",
        "run",
        "python",
        "-m",
        "tradingdev.mcp.workers.backtest",
        "--job-id",
        "job_123",
    ]
    assert calls[0][1]["cwd"] == str(tmp_path.resolve())
    assert calls[0][1]["start_new_session"] is True
