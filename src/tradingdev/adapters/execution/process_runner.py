"""Subprocess execution adapter."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


class ProcessRunner:
    """Spawn detached Python module subprocesses."""

    def __init__(self, project_root: Path | None = None) -> None:
        self._project_root = (project_root or Path.cwd()).resolve()

    def spawn_module(self, module: str, *args: str) -> int:
        """Run ``python -m <module>`` detached and return the PID."""
        proc = subprocess.Popen(  # noqa: S603
            [sys.executable, "-m", module, *args],
            cwd=str(self._project_root),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        return int(proc.pid)
