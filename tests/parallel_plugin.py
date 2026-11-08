"""Choose pytest concurrency without changing product execution policy."""

from __future__ import annotations

import os
import platform
import subprocess
import warnings
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest


@pytest.hookimpl(tryfirst=True)
def pytest_configure(config: pytest.Config) -> None:
    """Own xdist's default temporary root without sweeping other test runs."""
    if (
        hasattr(config, "workerinput")
        or config.getoption("dist", default="no") == "no"
        or not config.getoption("tx", default=[])
        or config.getoption("basetemp", default=None) is not None
    ):
        return
    directory = TemporaryDirectory(prefix="tradingdev-pytest-")
    config.add_cleanup(directory.cleanup)
    config.option.basetemp = str(Path(directory.name) / "workers")


@pytest.hookimpl(tryfirst=True, optionalhook=True)
def pytest_xdist_auto_num_workers(config: pytest.Config) -> int | None:
    """Honor explicit counts, then prefer Apple's highest performance tier."""
    mode = config.getoption("numprocesses")
    if mode not in {"auto", "logical"}:
        return None
    override = os.environ.get("PYTEST_XDIST_AUTO_NUM_WORKERS")
    if override is not None:
        try:
            count = int(override)
            if count >= 0:
                return count
        except ValueError:
            pass
        raise pytest.UsageError(
            "PYTEST_XDIST_AUTO_NUM_WORKERS must be a nonnegative integer "
            "(0 disables parallel execution)"
        )
    if mode != "auto":
        # An explicit -n logical retains xdist's logical CPU policy.
        return None
    system = platform.system()
    if system in {"Linux", "Windows"}:
        return max(1, (os.process_cpu_count() or 1) - 2)
    if system != "Darwin" or platform.machine() != "arm64":
        # Preserve xdist's physical CPU detection on Intel Macs/other hosts.
        return None
    try:
        result = subprocess.run(
            ["/usr/sbin/sysctl", "-n", "hw.perflevel0.physicalcpu"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        count = int(result.stdout.strip())
        if count > 0:
            return count
    except OSError, ValueError, subprocess.SubprocessError:
        pass
    warnings.warn(
        pytest.PytestConfigWarning(
            "Cannot determine macOS performance cores; using one pytest worker. "
            "Set PYTEST_XDIST_AUTO_NUM_WORKERS or pass -n to choose explicitly."
        ),
        stacklevel=2,
    )
    return 1
