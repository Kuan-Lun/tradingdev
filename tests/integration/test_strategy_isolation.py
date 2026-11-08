"""Real Linux container boundaries; requires a prebuilt strategy-runtime image."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from tempfile import mkdtemp
from typing import TYPE_CHECKING

import pytest

from tradingdev.adapters.execution.container_runtime import DockerRuntime
from tradingdev.adapters.execution.process_runner import ProcessIdentity
from tradingdev.domain.runtime import IsolatedRuntimeError, RuntimeLimits

if TYPE_CHECKING:
    from collections.abc import Iterator

_PYTHON = "/app/.venv/bin/python"


@pytest.fixture
def runtime_root() -> Iterator[Path]:
    """Retain ownership records if a daemon/client cannot be confirmed stopped."""
    directory = Path(mkdtemp(prefix="tradingdev-isolation-test-"))
    try:
        yield directory
    finally:
        remaining = list(directory.rglob("*.json"))
        if remaining:
            raise AssertionError(f"Unverified runtime cleanup; retained {directory}")
        shutil.rmtree(directory)


def _limits(**changes: object) -> RuntimeLimits:
    values: dict[str, object] = {
        "max_seconds": 5,
        "memory_bytes": 128 * 1024**2,
        "temporary_bytes": 1024**2,
        "cpus": 0.5,
        "processes": 24,
        "message_bytes": 16384,
        "diagnostic_bytes": 4096,
    }
    values.update(changes)
    return RuntimeLimits.model_validate(values)


def _wait_absent(runtime: DockerRuntime, owner: str) -> None:
    deadline = time.monotonic() + 10
    while True:
        result = runtime.command(
            "ps",
            "--all",
            "--quiet",
            "--filter",
            f"label=org.tradingdev.execution.owner={owner}",
        )
        assert result.returncode == 0, result.stderr
        if not result.stdout.strip():
            return
        assert time.monotonic() < deadline, "Owned container survived its deadline"
        time.sleep(0.05)


def test_kernel_enforces_identity_files_network_and_resource_limits(
    runtime_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TRADINGDEV_TEST_SECRET", "must-not-enter-the-worker")
    runtime = DockerRuntime(runtime_root / "controls")
    limits = _limits()
    spec = runtime.resolve(limits=limits)
    script = """
import errno, json, os, signal, socket, sys
from pathlib import Path
sys.stdin.readline()
report = {"uid": os.getuid(), "interfaces": os.listdir("/sys/class/net"),
          "secret": os.getenv("TRADINGDEV_TEST_SECRET"),
          "docker_socket": Path("/var/run/docker.sock").exists(),
          "memory": Path("/sys/fs/cgroup/memory.max").read_text().strip(),
          "pids": Path("/sys/fs/cgroup/pids.max").read_text().strip(),
          "cpu": Path("/sys/fs/cgroup/cpu.max").read_text().strip()}
with socket.socket() as connection:
    connection.settimeout(0.5)
    report["external_connection"] = errno.errorcode[
        connection.connect_ex(("192.0.2.1", 9))]
try:
    os.kill(1, signal.SIGTERM)
except PermissionError:
    report["init_protected"] = True
try:
    Path("/app/forbidden").write_text("forbidden")
except OSError as error:
    report["root_readonly"] = error.errno == errno.EROFS
try:
    Path("/tmp/too-large").write_bytes(b"x" * (2 * 1024**2))
except OSError as error:
    report["temporary_bounded"] = error.errno == errno.ENOSPC
report["capabilities"] = next(line.split()[1] for line in
    Path("/proc/self/status").read_text().splitlines() if line.startswith("CapEff:"))
print(json.dumps(report))
sys.stdin.readline()
"""
    with runtime.open(spec, (_PYTHON, "-I", "-u", "-c", script)) as worker:
        result = json.loads(worker.exchange(b"{}"))
        owner = worker.owner
        assert worker.container_id is not None
        network = runtime.command(
            "inspect", "--format", "{{.HostConfig.NetworkMode}}", worker.container_id
        )
        assert network.returncode == 0
        assert network.stdout.strip() == "none"
    assert result["uid"] == 65534
    assert result["capabilities"] == "0000000000000000"
    assert result["init_protected"]
    assert result["external_connection"] == "ENETUNREACH"
    assert result["secret"] is None
    assert result["docker_socket"] is False
    assert result["root_readonly"]
    assert result["temporary_bounded"]
    assert int(result["memory"]) == limits.memory_bytes
    assert int(result["pids"]) == limits.processes
    quota, period = map(int, result["cpu"].split())
    assert quota / period == pytest.approx(limits.cpus)
    _wait_absent(runtime, owner)
    assert not list(runtime.control_root.iterdir())


@pytest.mark.parametrize("outcome", ["normal", "failure", "timeout"])
def test_detached_session_is_removed_for_each_completion_path(
    runtime_root: Path, outcome: str
) -> None:
    runtime = DockerRuntime(runtime_root / "controls")
    spec = runtime.resolve(limits=_limits(max_seconds=3))
    script = f"""
import json, os, subprocess, sys, time
sys.stdin.readline()
child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"],
    start_new_session=True, stdin=subprocess.DEVNULL,
    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
os.kill(child.pid, 0)
print(json.dumps({{"pid": child.pid, "session": os.getsid(child.pid),
                  "parent_session": os.getsid(0)}}))
sys.stdin.readline()
if {outcome!r} == "failure":
    raise SystemExit(7)
if {outcome!r} == "timeout":
    time.sleep(60)
print('{{"finished": true}}')
"""
    failure: IsolatedRuntimeError | None = None
    try:
        with runtime.open(spec, (_PYTHON, "-I", "-u", "-c", script)) as worker:
            ready = json.loads(worker.exchange(b"start"))
            assert ready["pid"] == ready["session"]
            assert ready["session"] != ready["parent_session"]
            completed = json.loads(worker.exchange(b"finish"))
            assert completed == {"finished": True}
            # Normal exit must clean the namespace without the host close method.
            _wait_absent(runtime, worker.owner)
    except IsolatedRuntimeError as error:
        failure = error
    if outcome == "normal":
        assert failure is None
    else:
        assert failure is not None
        assert failure.code in {"runtime_exit", "runtime_timeout"}
        assert failure.cleanup_verified
    _wait_absent(runtime, worker.owner)
    assert not list(runtime.control_root.iterdir())


def test_process_limit_contains_forking_and_close_does_not_stop_another_unit(
    runtime_root: Path,
) -> None:
    runtime = DockerRuntime(runtime_root / "controls")
    spec = runtime.resolve(limits=_limits())
    script = """
import json, subprocess, sys, time
sys.stdin.readline()
children = []
try:
    for index in range(50):
        children.append(subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
except BlockingIOError:
    print(json.dumps({"bounded": True, "started": len(children)}))
time.sleep(60)
"""
    echo = "import sys\nfor line in sys.stdin: print(line.strip())"
    with runtime.open(spec, (_PYTHON, "-I", "-u", "-c", echo)) as unrelated:
        with runtime.open(spec, (_PYTHON, "-I", "-u", "-c", script)) as worker:
            result = json.loads(worker.exchange(b"start"))
            assert result["bounded"]
            assert 0 < result["started"] < spec.limits.processes
        _wait_absent(runtime, worker.owner)
        assert unrelated.exchange(b"still-alive") == b"still-alive"
    assert not list(runtime.control_root.iterdir())


def test_lost_host_controller_still_expires_and_control_record_is_recoverable(
    runtime_root: Path,
) -> None:
    runtime = DockerRuntime(runtime_root / "controls")
    spec = runtime.resolve(limits=_limits(max_seconds=3))
    controller = """
import json, os, sys
from pathlib import Path
from tradingdev.adapters.execution.container_runtime import DockerRuntime
from tradingdev.adapters.execution.process_runner import ProcessIdentity
from tradingdev.domain.runtime import RuntimeSpec
runtime = DockerRuntime(Path(sys.argv[1]))
context = runtime.open(RuntimeSpec.model_validate_json(sys.argv[2]),
    ("/app/.venv/bin/python", "-I", "-u", "-c",
     "import sys,time; sys.stdin.readline(); print('ready'); time.sleep(60)"))
worker = context.__enter__()
assert worker.exchange(b"start") == b"ready"
identity = ProcessIdentity.capture(worker._process.pid)
print(json.dumps({"owner": worker.owner, "pid": identity.pid,
                  "create_time": identity.create_time}), flush=True)
os._exit(0)
"""
    client: ProcessIdentity | None = None
    try:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                controller,
                str(runtime.control_root),
                spec.model_dump_json(),
            ],
            capture_output=True,
            text=True,
            timeout=20,
            check=True,
        )
        launched = json.loads(result.stdout)
        client = ProcessIdentity(launched["pid"], launched["create_time"])
        _wait_absent(runtime, launched["owner"])
        assert list(runtime.control_root.glob("*.json"))
    finally:
        runtime.recover_abandoned()
        if client is not None:
            client_record = runtime_root / "client-cleanup.json"
            client_record.write_text(
                json.dumps({"pid": client.pid, "create_time": client.create_time}),
                encoding="utf-8",
            )
            deadline = time.monotonic() + 5
            while client.get_process() is not None:
                assert time.monotonic() < deadline, (
                    "Docker attach client survived cleanup"
                )
                time.sleep(0.05)
            client_record.unlink()
    assert not list(runtime.control_root.iterdir())


def test_memory_exhaustion_is_a_failed_execution_with_verified_cleanup(
    runtime_root: Path,
) -> None:
    runtime = DockerRuntime(runtime_root / "controls")
    spec = runtime.resolve(limits=_limits(memory_bytes=64 * 1024**2))
    script = (
        "import sys; sys.stdin.readline(); "
        "bytearray(256 * 1024**2); print('unexpected')"
    )
    with (
        pytest.raises(IsolatedRuntimeError) as failure,
        runtime.open(spec, (_PYTHON, "-I", "-u", "-c", script)) as worker,
    ):
        worker.exchange(b"start")
    assert failure.value.code == "runtime_exit"
    assert failure.value.cleanup_verified
    _wait_absent(runtime, worker.owner)
    assert not list(runtime.control_root.iterdir())


@pytest.mark.parametrize("stream", [1, 2], ids=["stdout", "stderr"])
def test_undrained_output_does_not_block_explicit_cleanup(
    runtime_root: Path,
    stream: int,
) -> None:
    runtime = DockerRuntime(runtime_root / "controls")
    spec = runtime.resolve(limits=_limits(max_seconds=60))
    script = (
        "import os,sys; sys.stdin.readline(); print('ready'); "
        "sys.stdin.readline()\n"
        f"while True: os.write({stream}, b'x' * 65536)"
    )
    with runtime.open(spec, (_PYTHON, "-I", "-u", "-c", script)) as worker:
        assert worker.exchange(b"start") == b"ready"
        assert worker._process is not None and worker._process.stdin is not None
        worker._process.stdin.write(b"flood\n")
        worker._process.stdin.flush()
        time.sleep(0.5)
        started = time.monotonic()
    assert time.monotonic() - started < 10
    assert worker._process.returncode is not None
    _wait_absent(runtime, worker.owner)
    assert not list(runtime.control_root.iterdir())
