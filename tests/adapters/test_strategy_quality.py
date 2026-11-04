"""Quality-tool failures retain diagnostics without skipping the other tool."""

from __future__ import annotations

import subprocess
import sys
from typing import TYPE_CHECKING

import pytest

from tradingdev.adapters.execution.strategy_quality import run_quality_gates

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize("tool", ["ruff", "mypy"])
@pytest.mark.parametrize("failure", ["unavailable", "timeout", "stdout", "stderr"])
def test_quality_failure_keeps_tool_identity_and_checks_both_tools(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tool: str,
    failure: str,
) -> None:
    source, config = tmp_path / "strategy.py", tmp_path / "pyproject.toml"
    commands: list[list[str]] = []

    def run(command: list[str], **options: object) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        name = command[2]
        assert command[:2] == [sys.executable, "-m"]
        assert command[-1] == str(source)
        assert command[command.index(str(config)) - 1] == (
            "--config" if name == "ruff" else "--config-file"
        )
        assert options["timeout"] == (30 if name == "ruff" else 60)
        if name != tool:
            return subprocess.CompletedProcess(command, 0, "", "")
        if failure == "unavailable":
            raise FileNotFoundError("missing interpreter")
        if failure == "timeout":
            raise subprocess.TimeoutExpired(command, float(str(options["timeout"])))
        return subprocess.CompletedProcess(
            command,
            1,
            "invalid strategy" if failure == "stdout" else "",
            "invalid strategy" if failure == "stderr" else "ignored stderr",
        )

    monkeypatch.setattr(subprocess, "run", run)
    diagnostics = run_quality_gates(source, config)

    assert [command[2] for command in commands] == ["ruff", "mypy"]
    assert len(diagnostics) == 1
    result = diagnostics[0]
    suffix = "failed" if failure in {"stdout", "stderr"} else failure
    assert result.code == f"{tool}_{suffix}"
    assert result.phase == "quality_gate" and result.level == "error"
    assert result.fix
    if suffix == "failed":
        assert result.message == f"{tool} failed: invalid strategy"
