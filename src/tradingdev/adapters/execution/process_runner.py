"""Subprocess execution adapter."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from tradingdev.adapters.storage.filesystem import WorkspacePaths


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

    def spawn_module(self, module: str, *args: str) -> int:
        """Run a module in this interpreter and workspace; return its detached PID."""
        proc = subprocess.Popen(  # noqa: S603
            [sys.executable, "-m", module, *args],
            cwd=str(self._project_root),
            env=self._env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        return int(proc.pid)
