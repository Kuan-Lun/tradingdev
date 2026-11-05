"""Real supervisor recovery of generated units across host and child sessions."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import pytest
from tooling.process_guard import run_captured

from tests.integration.mcp_harness import temporary_mcp_workspace
from tradingdev.adapters.execution.container_runtime import DockerRuntime
from tradingdev.adapters.execution.process_runner import (
    BoundedWorkerError,
    ProcessRunner,
)
from tradingdev.adapters.storage.filesystem import WorkspacePaths

pytestmark = pytest.mark.integration

_STRATEGY = """
import os, subprocess, sys
from tradingdev.domain.strategies.base import BaseStrategy
class Detached(BaseStrategy):
    def generate_signals(self, df):
        self.child = subprocess.Popen(
            [sys.executable, '-c', 'import time; time.sleep(120)'],
            start_new_session=True, stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        result = df.copy()
        result['signal'] = 0
        return result
    def get_parameters(self):
        return {'child_pid': self.child.pid, 'child_session': os.getsid(self.child.pid),
                'worker_session': os.getsid(0)}
"""

_PROBE = """
import hashlib, json, os, sys, time
from pathlib import Path
import pandas as pd
from tradingdev.adapters.execution.strategy_factory import StrategyFactory
root = Path.cwd()
path = root / 'generated_strategies/detached.py'
config = {'strategy': {'id': 'detached', 'class_name': 'Detached',
    'source_path': str(path),
    'source_hash': hashlib.sha256(path.read_bytes()).hexdigest(),
    'parameters': {}}}
factory = StrategyFactory(workspace_root=root)
with factory.open_from_config(config, None) as strategy:
    strategy.generate_signals(pd.DataFrame({'close': [1.0]}))
    controls = Path(os.environ['TRADINGDEV_RUNTIME_CONTROL_ROOT'])
    records = list(controls.glob('*.json'))
    assert len(records) == 1
    record = json.loads(records[0].read_text())
    record['parameters'] = strategy.get_parameters()
    (root / 'ready.tmp').write_text(json.dumps(record))
    (root / 'ready.tmp').replace(root / 'ready.json')
    if sys.argv[1] == 'abrupt':
        os._exit(7)
    time.sleep(120)
"""


def _read_ready(path: Path) -> dict[str, Any]:
    deadline = time.monotonic() + 20
    while not path.exists():
        if time.monotonic() >= deadline:
            raise AssertionError("Generated unit did not reach the cleanup probe")
        time.sleep(0.05)
    value: dict[str, Any] = json.loads(path.read_text())
    return value


@pytest.mark.parametrize("outcome", ["cancel_other_session", "timeout", "abrupt"])
def test_supervisor_removes_container_with_detached_generated_descendant(
    monkeypatch: pytest.MonkeyPatch, outcome: str
) -> None:
    with temporary_mcp_workspace() as workspace:
        root = workspace.root
        source = Path(__file__).resolve().parents[2] / "src"
        monkeypatch.setenv("PYTHONPATH", os.pathsep.join((str(root), str(source))))
        (root / "generated_strategies").mkdir()
        (root / "generated_strategies/detached.py").write_text(_STRATEGY)
        (root / "cleanup_probe.py").write_text(_PROBE)
        runner = ProcessRunner(root, workspace=WorkspacePaths(workspace.workspace))
        if outcome == "timeout":
            with pytest.raises(BoundedWorkerError) as failure:
                runner.run_module("cleanup_probe", outcome, timeout_seconds=10)
            assert failure.value.code == "preflight_timeout"
            assert failure.value.cleanup_verified
        else:
            handle = runner.spawn_module("cleanup_probe", outcome)
            _read_ready(root / "ready.json")
            # A separate native process knows only the persisted launch identity.
            # It has no Popen handle or inherited relationship to the supervisor.
            stopped = run_captured(
                [
                    sys.executable,
                    "-c",
                    "import json, sys; from pathlib import Path; "
                    "from tradingdev.adapters.execution.process_runner import "
                    "WorkerHandle, request_worker_stop; "
                    "handle = WorkerHandle.from_job(json.loads(sys.argv[2])); "
                    "assert handle is not None; "
                    "request_worker_stop(Path(sys.argv[1]), handle)",
                    str(workspace.workspace),
                    json.dumps(handle.job_fields()),
                ],
                cwd=root,
                env=dict(os.environ),
                timeout=20,
            )
            assert stopped.returncode == 0, stopped.stderr
            assert handle.get_process() is None
        ready = _read_ready(root / "ready.json")
        params = ready["parameters"]
        assert params["child_pid"] == params["child_session"]
        assert params["child_session"] != params["worker_session"]
        controls = list((workspace.workspace / ".workers").iterdir())
        assert len(controls) == 1
        finished = json.loads((controls[0] / "finished.json").read_text())
        assert finished["cleaned"] is True, finished
        if outcome == "abrupt":
            assert finished["returncode"] == 7
        assert not list((controls[0] / "runtime").glob("*.json"))
        runtime = DockerRuntime(controls[0] / "runtime")
        remaining = runtime.command(
            "ps",
            "--all",
            "--quiet",
            "--filter",
            f"label=org.tradingdev.execution.owner={ready['owner']}",
        )
        assert remaining.returncode == 0, remaining.stderr
        assert not remaining.stdout.strip()
    assert not root.exists()
