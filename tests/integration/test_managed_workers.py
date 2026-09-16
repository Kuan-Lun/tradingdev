"""Real supervisor cleanup, including descendants of an already-exited leader."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import psutil
import pytest

from tests.integration.mcp_harness import temporary_mcp_workspace
from tradingdev.adapters.execution.process_runner import (
    ProcessIdentity,
    ProcessRunner,
    request_worker_stop,
)
from tradingdev.adapters.storage.filesystem import WorkspacePaths

pytestmark = pytest.mark.integration

_PROBE = '''
import json, os, signal, subprocess, sys, time
from pathlib import Path
import psutil
signal.signal(signal.SIGTERM, signal.SIG_IGN)
child_code = """
import json, os, signal, time
from pathlib import Path
import psutil
signal.signal(signal.SIGTERM, signal.SIG_IGN)
Path('descendant.json').write_text(json.dumps({
    'pid': os.getpid(), 'created': psutil.Process().create_time()
}))
time.sleep(60)
"""
child = subprocess.Popen([sys.executable, '-c', child_code])
while not Path('descendant.json').exists():
    time.sleep(0.01)
Path('leader.json').write_text(json.dumps({
    'pid': os.getpid(), 'created': psutil.Process().create_time()
}))
if sys.argv[1] == 'natural':
    sys.exit(0)
if sys.argv[1] == 'failure':
    sys.exit(7)
time.sleep(60)
'''


def _wait_json(path: Path) -> dict[str, int | float]:
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if path.exists():
            try:
                value = json.loads(path.read_text())
                assert isinstance(value, dict)
                return value
            except json.JSONDecodeError:
                pass
        time.sleep(0.02)
    raise AssertionError(f"Worker did not write {path}")


def _set_module_path(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = Path(__file__).resolve().parents[2] / "src"
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join((str(root), str(source))))


@pytest.mark.parametrize("exit_mode", ["natural", "failure", "cancel"])
def test_supervisor_stops_descendants_before_acknowledging_cleanup(
    monkeypatch: pytest.MonkeyPatch, exit_mode: str
) -> None:
    with temporary_mcp_workspace() as workspace:
        root = workspace.root
        _set_module_path(root, monkeypatch)
        (root / "probe_worker.py").write_text(_PROBE)
        runner = ProcessRunner(root, workspace=WorkspacePaths(workspace.workspace))
        handle = runner.spawn_module("probe_worker", exit_mode)
        leader = _wait_json(root / "leader.json")
        descendant = _wait_json(root / "descendant.json")
        control = workspace.workspace / ".workers" / handle.control_id
        if exit_mode != "cancel":
            # Let the supervisor observe natural leader exit, without a stop
            # request. The TERM-resistant descendant remains in that group.
            _wait_json(control / "finished.json")
        request_worker_stop(workspace.workspace, handle)
        finished = json.loads((control / "finished.json").read_text())
        assert finished["cleaned"] is True
        if exit_mode != "cancel":
            assert finished["returncode"] == (0 if exit_mode == "natural" else 7)
        for record in (leader, descendant):
            identity = ProcessIdentity.from_values(record["pid"], record["created"])
            assert identity is not None
            assert identity.get_process() is None
    assert not root.exists()


def test_identity_capture_failure_cannot_start_a_detached_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[ProcessIdentity] = []
    capture = ProcessIdentity.capture

    def fail_capture(pid: int) -> ProcessIdentity:
        captured.append(capture(pid))
        raise psutil.AccessDenied(pid)

    with temporary_mcp_workspace() as workspace:
        root = workspace.root
        _set_module_path(root, monkeypatch)
        (root / "probe_worker.py").write_text(_PROBE)
        monkeypatch.setattr(ProcessIdentity, "capture", fail_capture)
        runner = ProcessRunner(root, workspace=WorkspacePaths(workspace.workspace))
        with pytest.raises(psutil.AccessDenied):
            runner.spawn_module("probe_worker", "cancel")
        assert len(captured) == 1
        assert captured[0].get_process() is None
        assert not (root / "leader.json").exists()
        assert not (root / "descendant.json").exists()
        assert not list((workspace.workspace / ".workers").iterdir())
    assert not root.exists()
