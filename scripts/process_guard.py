"""Run a check with bounded cleanup of its observed subprocess identities."""

from __future__ import annotations

import os
import signal
import subprocess
import time
from contextlib import suppress
from tempfile import TemporaryFile
from typing import TYPE_CHECKING

import psutil

if TYPE_CHECKING:
    from pathlib import Path
    from typing import TextIO

_GRACE_SECONDS = 5.0
_STOP_SECONDS = 3.0
_POLL_SECONDS = 0.02


def _alive(process: psutil.Process) -> bool:
    try:
        return bool(process.is_running() and process.status() != psutil.STATUS_ZOMBIE)
    except psutil.NoSuchProcess:
        return False
    except (psutil.AccessDenied, PermissionError):
        return bool(process.is_running())


def _observe(known: set[psutil.Process]) -> bool:
    complete = True
    for process in tuple(known):
        if _alive(process):
            try:
                known.update(process.children(recursive=True))
            except psutil.NoSuchProcess:
                pass
            except (psutil.AccessDenied, PermissionError):
                complete = False
    return complete


def _signal_leader(leader: psutil.Process | None, sig: signal.Signals) -> None:
    if leader is None or not _alive(leader):
        return
    try:
        if os.getpgid(leader.pid) == leader.pid:
            os.killpg(leader.pid, sig)
            return
    except (ProcessLookupError, PermissionError):
        pass
    with suppress(psutil.NoSuchProcess, psutil.AccessDenied, PermissionError):
        leader.send_signal(sig)


def _cleanup(leader: psutil.Process | None, known: set[psutil.Process]) -> None:
    inspection_complete = _observe(known)
    for sig in (signal.SIGTERM, signal.SIGKILL):
        if not any(_alive(process) for process in known):
            break
        _signal_leader(leader, sig)
        for process in known:
            if _alive(process):
                with suppress(
                    psutil.NoSuchProcess, psutil.AccessDenied, PermissionError
                ):
                    process.send_signal(sig)
        deadline = time.monotonic() + _STOP_SECONDS
        while any(_alive(process) for process in known):
            inspection_complete = _observe(known) and inspection_complete
            if time.monotonic() >= deadline:
                break
            time.sleep(_POLL_SECONDS)
    survivors = sorted(process.pid for process in known if _alive(process))
    if survivors:
        raise RuntimeError(f"Could not terminate check subprocesses: {survivors}")
    if not inspection_complete:
        raise RuntimeError(
            "Process inspection was denied; descendant cleanup is unverified"
        )


def _run(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    timeout: float,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    if not _observe({psutil.Process()}):
        raise RuntimeError(
            "Process inspection access is required before running checks"
        )
    process = subprocess.Popen(
        command, cwd=cwd, env=env, stdout=stdout, stderr=stderr, start_new_session=True
    )
    leader: psutil.Process | None = None
    known: set[psutil.Process] = set()
    failure: BaseException | None = None
    try:
        with suppress(psutil.NoSuchProcess):
            leader = psutil.Process(process.pid)
            known.add(leader)
        deadline = time.monotonic() + timeout
        while process.poll() is None:
            if not _observe(known):
                raise RuntimeError(
                    "Process inspection access is required to track check subprocesses"
                )
            if time.monotonic() >= deadline:
                raise subprocess.TimeoutExpired(command, timeout)
            time.sleep(_POLL_SECONDS)
    except BaseException as error:
        failure = error
        if isinstance(error, (subprocess.TimeoutExpired, KeyboardInterrupt)):
            _signal_leader(leader, signal.SIGINT)
            deadline = time.monotonic() + _GRACE_SECONDS
            while process.poll() is None and time.monotonic() < deadline:
                _observe(known)
                time.sleep(_POLL_SECONDS)
    finally:
        try:
            _cleanup(leader, known)
        except RuntimeError as error:
            if failure is not None:
                raise RuntimeError(
                    f"{type(failure).__name__}: {failure}; cleanup failed: {error}"
                ) from failure
            raise
        finally:
            if process.poll() is not None:
                process.wait()
    if failure is not None:
        raise failure
    assert process.returncode is not None
    return process.returncode


def run_checked(
    command: list[str], *, cwd: Path, env: dict[str, str], timeout: float
) -> None:
    """Run a check, allowing teardown on interruption before stopping descendants."""
    returncode = _run(command, cwd=cwd, env=env, timeout=timeout)
    if returncode:
        raise subprocess.CalledProcessError(returncode, command)


def run_captured(
    command: list[str], *, cwd: Path, env: dict[str, str], timeout: float
) -> subprocess.CompletedProcess[str]:
    """Capture output without pipe backpressure, with the same descendant cleanup."""
    with (
        TemporaryFile(mode="w+", encoding="utf-8", errors="replace") as stdout,
        TemporaryFile(mode="w+", encoding="utf-8", errors="replace") as stderr,
    ):
        try:
            returncode = _run(
                command,
                cwd=cwd,
                env=env,
                timeout=timeout,
                stdout=stdout,
                stderr=stderr,
            )
        except subprocess.TimeoutExpired as error:
            stdout.seek(0)
            stderr.seek(0)
            error.output = stdout.read().encode("utf-8")
            error.stderr = stderr.read().encode("utf-8")
            raise
        stdout.seek(0)
        stderr.seek(0)
        return subprocess.CompletedProcess(
            command, returncode, stdout.read(), stderr.read()
        )
