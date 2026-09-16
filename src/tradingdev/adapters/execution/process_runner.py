"""Subprocess execution adapter."""

from __future__ import annotations

import math
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import psutil

from tradingdev.adapters.storage.filesystem import WorkspacePaths


@dataclass(frozen=True)
class ProcessIdentity:
    """Persisted identity of one process, distinct from a reusable PID."""

    pid: int
    create_time: float

    @classmethod
    def capture(cls, pid: int) -> ProcessIdentity:
        """Capture creation time while the caller still owns the process."""
        return cls(pid, float(psutil.Process(pid).create_time()))

    @classmethod
    def from_values(cls, pid: object, create_time: object) -> ProcessIdentity | None:
        """Reject absent or invalid identities without guessing from a PID."""
        if (
            not isinstance(pid, int)
            or isinstance(pid, bool)
            or pid <= 0
            or not isinstance(create_time, int | float)
            or isinstance(create_time, bool)
            or not math.isfinite(create_time)
            or create_time <= 0
        ):
            return None
        return cls(pid, float(create_time))

    def get_process(self) -> psutil.Process | None:
        """Resolve the original live process; permission failures propagate."""
        try:
            process = psutil.Process(self.pid)
            if (
                process.create_time() != self.create_time
                or not process.is_running()
                or process.status() == psutil.STATUS_ZOMBIE
            ):
                return None
            return process
        except psutil.NoSuchProcess:
            return None


class ProcessRunner:
    """Spawn detached Python module subprocesses."""

    def __init__(
        self,
        project_root: Path | None = None,
        *,
        workspace: WorkspacePaths | None = None,
    ) -> None:
        configured_root = os.environ.get("TRADINGDEV_PROJECT_ROOT")
        self._project_root = (
            (project_root or (Path(configured_root) if configured_root else Path.cwd()))
            .expanduser()
            .resolve()
        )
        self._env = os.environ.copy()
        self._env["TRADINGDEV_WORKSPACE"] = str((workspace or WorkspacePaths()).root)
        if data_root := self._env.get("TRADINGDEV_DATA_ROOT"):
            self._env["TRADINGDEV_DATA_ROOT"] = str(
                Path(data_root).expanduser().resolve()
            )

    def spawn_module(self, module: str, *args: str) -> ProcessIdentity:
        """Run a detached module and capture its identity before releasing it."""
        proc = subprocess.Popen(  # noqa: S603
            [sys.executable, "-m", module, *args],
            cwd=str(self._project_root),
            env=self._env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        try:
            return ProcessIdentity.capture(proc.pid)
        except BaseException:
            # Keep the owned Popen handle until capture succeeds. If it fails,
            # reap this child rather than leave an untracked detached worker.
            proc.kill()
            proc.wait(timeout=5)
            raise
