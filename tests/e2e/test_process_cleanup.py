"""Regression checks for restricted macOS process-group cleanup."""

from __future__ import annotations

import os
import select
import signal
import subprocess
import sys
from typing import TYPE_CHECKING

import psutil
import pytest

from tests.e2e.codex_harness import _process_identity, _terminate_tree

if TYPE_CHECKING:
    from pytest import MonkeyPatch


def _sleeper() -> subprocess.Popen[bytes]:
    return subprocess.Popen(  # noqa: S603
        [sys.executable, "-c", "import time; time.sleep(60)"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


def test_cleanup_does_not_signal_a_process_group_after_its_leader_exits(
    monkeypatch: MonkeyPatch,
) -> None:
    process = _sleeper()
    identity = _process_identity(process.pid)
    assert identity is not None
    process.terminate()
    process.wait(timeout=3)

    def reject_signal(pid: int, sig: int) -> None:
        pytest.fail("Cleanup signalled an exited process group")

    monkeypatch.setattr(os, "killpg", reject_signal)
    _terminate_tree(identity)


def test_group_permission_error_falls_back_to_individual_process_signals(
    monkeypatch: MonkeyPatch,
) -> None:
    process = _sleeper()
    try:
        identity = _process_identity(process.pid)
        assert identity is not None
        signals: list[int] = []

        def deny_group(pid: int, sig: int) -> None:
            signals.append(sig)
            raise PermissionError("Process group signalling is denied")

        monkeypatch.setattr(os, "killpg", deny_group)
        _terminate_tree(identity)
        assert process.wait(timeout=3) == -signal.SIGTERM
        assert signals == [signal.SIGTERM]
    finally:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=3)


def test_cleanup_reports_known_processes_that_remain_alive(
    monkeypatch: MonkeyPatch,
) -> None:
    process = _sleeper()
    try:
        identity = _process_identity(process.pid)
        assert identity is not None

        def deny_group(pid: int, sig: int) -> None:
            raise PermissionError("Process group signalling is denied")

        def deny_individual(sig: int) -> None:
            raise psutil.AccessDenied(process.pid)

        monkeypatch.setattr(os, "killpg", deny_group)
        monkeypatch.setattr(identity, "send_signal", deny_individual)
        with pytest.raises(RuntimeError, match=f"subprocesses:.*{process.pid}"):
            _terminate_tree(identity, wait_seconds=0.01)
        assert process.poll() is None
    finally:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=3)


def test_cleanup_terminates_tracked_children_after_the_leader_exits(
    monkeypatch: MonkeyPatch,
) -> None:
    leader = _sleeper()
    leader_identity = _process_identity(leader.pid)
    leader.terminate()
    leader.wait(timeout=3)
    tracked = _sleeper()
    try:
        tracked_identity = _process_identity(tracked.pid)
        assert tracked_identity is not None

        def reject_group(pid: int, sig: int) -> None:
            pytest.fail("Cleanup signalled the exited leader's process group")

        monkeypatch.setattr(os, "killpg", reject_group)
        _terminate_tree(leader_identity, {tracked_identity})
        assert tracked.wait(timeout=3) == -signal.SIGTERM
    finally:
        if tracked.poll() is None:
            tracked.kill()
        tracked.wait(timeout=3)


def test_cleanup_escalates_when_a_process_ignores_sigterm(
    monkeypatch: MonkeyPatch,
) -> None:
    process = subprocess.Popen(  # noqa: S603
        [
            sys.executable,
            "-u",
            "-c",
            "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
            "print('ready'); time.sleep(60)",
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    try:
        assert process.stdout is not None
        assert select.select([process.stdout], [], [], 30)[0], (
            "Child never became ready"
        )
        assert process.stdout.readline() == b"ready\n"
        identity = _process_identity(process.pid)
        assert identity is not None
        signals: list[int] = []

        def deny_group(pid: int, sig: int) -> None:
            signals.append(sig)
            raise PermissionError("Process group signalling is denied")

        monkeypatch.setattr(os, "killpg", deny_group)
        _terminate_tree(identity, wait_seconds=0.01)
        assert process.wait(timeout=3) == -signal.SIGKILL
        assert signals == [signal.SIGTERM, signal.SIGKILL]
    finally:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=3)
        if process.stdout is not None:
            process.stdout.close()
