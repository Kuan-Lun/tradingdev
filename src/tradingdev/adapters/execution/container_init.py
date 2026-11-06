"""Own a private Linux PID namespace without executing generated code as PID 1.

The worker runs under another UID with no effective capabilities. The trusted
namespace init owns the wall-clock deadline even if its host client disappears.
Exiting PID 1 makes the kernel terminate every member of this PID namespace,
including descendants that have changed session or process group.
"""

from __future__ import annotations

import argparse
import math
import os
import signal
import subprocess
import sys
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import FrameType


def supervise(command: list[str], timeout: float) -> int:
    """Return the owned child's status, or terminate the whole namespace."""
    if sys.platform != "linux" or os.getpid() != 1 or os.geteuid() != 0:
        raise RuntimeError("The execution init requires root PID 1 in Linux")
    if not command or not math.isfinite(timeout) or not 0 < timeout <= 86400:
        raise ValueError("An execution command and a bounded deadline are required")
    stopped = False

    def stop(_signal: int, _frame: FrameType | None) -> None:
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    deadline = time.monotonic() + timeout
    worker = subprocess.Popen(
        command,
        user=65534,
        group=65534,
        extra_groups=[],
        umask=0o077,
    )
    while not stopped:
        if time.monotonic() >= deadline:
            # The worker shares these pipes and may have filled them. Never let
            # diagnostic I/O delay PID 1's unconditional namespace termination.
            return 124
        # Reap orphaned descendants too. Popen must not race with this sole waiter.
        while True:
            try:
                pid, status = os.waitpid(-1, os.WNOHANG)
            except ChildProcessError:
                return 125
            if pid == 0:
                break
            if pid == worker.pid:
                worker.returncode = os.waitstatus_to_exitcode(status)
                return (
                    worker.returncode
                    if worker.returncode >= 0
                    else 128 - worker.returncode
                )
        time.sleep(0.02)
    return 143


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout", type=float, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command
    if command[:1] == ["--"]:
        command = command[1:]
    # Do not wait for Python finalizers or join untrusted descendants on exit.
    os._exit(supervise(command, args.timeout))


if __name__ == "__main__":
    main()
