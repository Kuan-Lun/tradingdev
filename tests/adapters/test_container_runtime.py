"""Host transport and uncertain cleanup, with a fake daemon and real pipes."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest

from tradingdev.adapters.execution import container_init
from tradingdev.adapters.execution.container_runtime import (
    ContainerSession,
    DockerRuntime,
)
from tradingdev.domain.runtime import IsolatedRuntimeError, RuntimeLimits, RuntimeSpec

if TYPE_CHECKING:
    from pathlib import Path

_CONTAINER = "a" * 64
_SPEC = RuntimeSpec(image_id="sha256:" + "b" * 64)


def test_namespace_deadline_does_not_depend_on_writable_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BlockedOutput:
        def write(self, _value: str) -> None:
            raise BlockingIOError("The worker filled the shared diagnostic pipe")

        def flush(self) -> None:
            raise BlockingIOError("The worker filled the shared diagnostic pipe")

    now = 0.0

    def start(*_args: Any, **_kwargs: Any) -> SimpleNamespace:
        nonlocal now
        now = 1.0
        return SimpleNamespace(pid=2)

    with monkeypatch.context() as patch:
        patch.setattr(sys, "platform", "linux")
        patch.setattr(os, "getpid", lambda: 1)
        patch.setattr(os, "geteuid", lambda: 0)
        patch.setattr(signal, "signal", lambda *_args: None)
        patch.setattr(time, "monotonic", lambda: now)
        patch.setattr(sys, "stderr", BlockedOutput())
        patch.setattr(subprocess, "Popen", start)
        result = container_init.supervise(["bounded-worker"], 0.5)
    assert result == 124


def test_endpoint_is_frozen_and_remote_transports_are_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("DOCKER_CONTEXT", raising=False)
    monkeypatch.setenv("DOCKER_HOST", "unix:///original/docker.sock")
    runtime = DockerRuntime(tmp_path)
    calls: list[tuple[list[str], dict[str, str]]] = []

    def run(arguments: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append((arguments, kwargs["env"]))
        return subprocess.CompletedProcess(arguments, 0, "", "")

    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setenv("DOCKER_CONTEXT", "changed-context")
    monkeypatch.setenv("DOCKER_HOST", "ssh://remote")
    runtime.command("ps")
    assert calls[0][0] == ["docker", "--host", "unix:///original/docker.sock", "ps"]
    assert "DOCKER_CONTEXT" not in calls[0][1]
    assert calls[0][1]["DOCKER_HOST"] == "unix:///original/docker.sock"
    with pytest.raises(IsolatedRuntimeError, match="local Docker Unix socket"):
        DockerRuntime(tmp_path, endpoint="ssh://remote")
    assert len(calls) == 1


def test_explicit_context_takes_precedence_over_environment_host(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DOCKER_CONTEXT", "desktop-linux")
    monkeypatch.setenv("DOCKER_HOST", "ssh://ignored-host")
    calls: list[list[str]] = []

    def run(arguments: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append(arguments)
        return subprocess.CompletedProcess(arguments, 0, "unix:///desktop.sock\n", "")

    monkeypatch.setattr(subprocess, "run", run)
    DockerRuntime(tmp_path).command("ps")
    assert calls[0][1:4] == ["context", "inspect", "desktop-linux"]
    assert calls[1] == ["docker", "--host", "unix:///desktop.sock", "ps"]


class _Daemon(DockerRuntime):
    def __init__(self, root: Path) -> None:
        super().__init__(root, endpoint="unix:///test/docker.sock")
        self.owned = False
        self.unavailable = False
        self.capabilities: dict[str, object] = {
            "ID": "test-daemon",
            "OSType": "linux",
            "CgroupVersion": "2",
            "MemoryLimit": True,
            "CpuCfsQuota": True,
            "PidsLimit": True,
        }

    def command(self, *arguments: str) -> subprocess.CompletedProcess[str]:
        if self.unavailable:
            return subprocess.CompletedProcess(arguments, 1, "", "daemon unavailable")
        if arguments[0] == "info":
            return subprocess.CompletedProcess(
                arguments, 0, json.dumps(self.capabilities), ""
            )
        if arguments[:2] == ("image", "inspect"):
            return subprocess.CompletedProcess(arguments, 0, _SPEC.image_id, "")
        if arguments[0] == "create":
            self.owned = True
            return subprocess.CompletedProcess(arguments, 0, _CONTAINER, "")
        if arguments[0] == "ps":
            return subprocess.CompletedProcess(
                arguments, 0, _CONTAINER if self.owned else "", ""
            )
        if arguments[0] == "rm":
            self.owned = False
            return subprocess.CompletedProcess(arguments, 0, _CONTAINER, "")
        raise AssertionError(arguments)


def test_resolve_uses_daemon_json_capabilities_and_immutable_image(
    tmp_path: Path,
) -> None:
    daemon = _Daemon(tmp_path)
    spec = daemon.resolve(image="mutable-tag")
    assert spec.image_id == _SPEC.image_id
    assert not list(tmp_path.iterdir())
    daemon.capabilities["CpuCfsQuota"] = False
    with pytest.raises(IsolatedRuntimeError, match="cgroup v2") as failure:
        daemon.resolve(image="mutable-tag")
    assert failure.value.cleanup_verified
    assert not daemon.owned


def _client(monkeypatch: pytest.MonkeyPatch, script: str) -> None:
    original = subprocess.Popen

    def start(_command: list[str], **kwargs: Any) -> subprocess.Popen[bytes]:
        process = original([sys.executable, "-u", "-c", script], **kwargs)
        assert os.getpgid(process.pid) == os.getpgrp()
        return process

    monkeypatch.setattr(
        "tradingdev.adapters.execution.container_runtime.subprocess.Popen", start
    )


def test_transport_preserves_sequential_messages_and_removes_control(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _client(monkeypatch, "import sys\nfor line in sys.stdin: print(line.strip()[::-1])")
    daemon = _Daemon(tmp_path)
    with daemon.open(_SPEC, ("worker",)) as worker:
        assert worker.exchange(b"first") == b"tsrif"
        assert worker.exchange(b"second") == b"dnoces"
        control = worker.control_path
        assert control.is_file()
    assert not daemon.owned
    assert not control.exists()


def test_message_limit_excludes_the_protocol_delimiter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _client(monkeypatch, "import sys\nfor line in sys.stdin: print(line.strip())")
    daemon = _Daemon(tmp_path)
    limits = RuntimeLimits(message_bytes=4096, diagnostic_bytes=1024)
    with daemon.open(
        _SPEC.model_copy(update={"limits": limits}), ("worker",)
    ) as worker:
        assert worker.exchange(b"x" * 4096) == b"x" * 4096
        with pytest.raises(ValueError, match="bounded line"):
            worker.exchange(b"x" * 4097)
    assert not daemon.owned


def test_active_lease_and_identity_collision_cannot_remove_another_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _client(monkeypatch, "import sys\nfor line in sys.stdin: print(line.strip())")
    monkeypatch.setattr(
        "tradingdev.adapters.execution.container_runtime.uuid4",
        lambda: SimpleNamespace(hex="e" * 32),
    )
    daemon = _Daemon(tmp_path)
    with daemon.open(_SPEC, ("worker",)) as first:
        daemon.recover_abandoned()
        assert first.control_path.exists()
        with (
            pytest.raises(IsolatedRuntimeError, match="already in use"),
            daemon.open(_SPEC, ("worker",)),
        ):
            pytest.fail("An existing identity must never be replaced")
        assert first.exchange(b"original") == b"original"
        assert daemon.owned
    assert not daemon.owned
    assert not list(tmp_path.iterdir())


def test_daemon_change_cannot_be_reported_as_successful_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _client(monkeypatch, "import sys\nfor line in sys.stdin: print(line.strip())")
    daemon = _Daemon(tmp_path)
    with (
        pytest.raises(IsolatedRuntimeError, match="Docker daemon changed"),
        daemon.open(_SPEC, ("worker",)) as worker,
    ):
        assert worker.exchange(b"ready") == b"ready"
        daemon.capabilities["ID"] = "another-daemon"
    try:
        assert worker.control_path.exists()
        assert daemon.owned
    finally:
        daemon.capabilities["ID"] = "test-daemon"
        worker.close()
    assert not list(tmp_path.iterdir())


def test_failure_keeps_bounded_final_diagnostics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _client(
        monkeypatch,
        "import sys; sys.stdin.readline(); "
        "sys.stderr.write('x' * 5000 + 'LATEST'); "
        "sys.stderr.flush(); raise SystemExit(7)",
    )
    daemon = _Daemon(tmp_path)
    limits = RuntimeLimits(message_bytes=4096, diagnostic_bytes=1024)
    with (
        pytest.raises(IsolatedRuntimeError) as failure,
        daemon.open(_SPEC.model_copy(update={"limits": limits}), ("worker",)) as worker,
    ):
        worker.exchange(b"request")
    assert "earlier diagnostics truncated" in str(failure.value)
    assert str(failure.value).endswith("LATEST")
    assert len(str(failure.value)) < 1200
    assert failure.value.cleanup_verified


@pytest.mark.parametrize(
    ("script", "message", "code"),
    [
        ("import time; time.sleep(30)", b"x" * (1024**2), "runtime_timeout"),
        (
            "import sys; sys.stdin.readline(); print('x' * 5000)",
            b"input",
            "runtime_output_limit",
        ),
        ("raise SystemExit(7)", b"input", "runtime_exit"),
    ],
    ids=["blocked-input", "output-limit", "system-exit"],
)
def test_transport_bounds_blocked_input_output_and_abnormal_exit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    script: str,
    message: bytes,
    code: str,
) -> None:
    _client(monkeypatch, script)
    daemon = _Daemon(tmp_path)
    limits = RuntimeLimits(
        max_seconds=0.2 if code == "runtime_timeout" else 5,
        message_bytes=2 * 1024**2 if code == "runtime_timeout" else 4096,
        diagnostic_bytes=1024,
    )
    with (
        pytest.raises(IsolatedRuntimeError) as failure,
        daemon.open(_SPEC.model_copy(update={"limits": limits}), ("worker",)) as worker,
    ):
        worker.exchange(message)
    assert failure.value.code == code
    assert failure.value.cleanup_verified
    assert not daemon.owned
    assert not list(tmp_path.iterdir())


def test_cleanup_failure_retains_identity_and_can_be_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _client(monkeypatch, "import sys\nfor line in sys.stdin: print('ok')")
    daemon = _Daemon(tmp_path)
    with (
        pytest.raises(IsolatedRuntimeError) as failure,
        daemon.open(_SPEC, ("worker",)) as worker,
    ):
        assert worker.exchange(b"request") == b"ok"
        daemon.unavailable = True
    assert failure.value.code == "runtime_cleanup_failed"
    assert not failure.value.cleanup_verified
    assert worker.control_path.is_file()
    assert daemon.owned
    daemon.unavailable = False
    worker.close()
    assert not worker.control_path.exists()
    assert not daemon.owned


def test_interruption_still_cleans_the_owned_unit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _client(monkeypatch, "import time; time.sleep(30)")
    daemon = _Daemon(tmp_path)
    with pytest.raises(KeyboardInterrupt), daemon.open(_SPEC, ("worker",)):
        raise KeyboardInterrupt
    assert not daemon.owned
    assert not list(tmp_path.iterdir())


def test_unacknowledged_create_is_not_claimed_clean_when_listing_is_empty(
    tmp_path: Path,
) -> None:
    daemon = _Daemon(tmp_path)
    worker = ContainerSession(daemon, _SPEC, ("worker",))
    worker._record()
    worker._create_attempted = True
    with pytest.raises(IsolatedRuntimeError, match="not acknowledged"):
        worker.close()
    assert worker.control_path.is_file()
    # A delayed create response must remain attributable after the first cleanup.
    daemon.owned = True
    worker.close()
    assert not worker.control_path.exists()
    assert not daemon.owned
