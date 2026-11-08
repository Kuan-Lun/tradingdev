"""Run a requested full suite; clean checkouts also produce snapshot evidence."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

from filelock import FileLock, Timeout

from tooling.environment import EnvironmentCheckError, check_environment
from tooling.git.candidates import check_test_snapshot, execution_environment, git
from tooling.git.receipts import receipt_path
from tooling.process_guard import (
    ProcessCleanupError,
    report_cleanup_failure,
    run_checked,
)


def run_suite(profile: str, timeout_seconds: int) -> None:
    """Explicit suite requests always run quality checks and the entire suite."""
    deadline = time.monotonic() + timeout_seconds
    check_environment(profile)
    environment = execution_environment(Path.cwd())
    run_checked(
        ["bash", "scripts/check-fast.sh"],
        cwd=Path.cwd(),
        env=environment,
        timeout=max(0, deadline - time.monotonic()),
    )
    run_checked(
        [
            sys.executable,
            "-m",
            "pytest",
            *(["tests/tooling"] if profile == "tooling" else []),
            "-n",
            "auto",
            "--dist",
            "worksteal",
            "--max-worker-restart=0",
        ],
        cwd=Path.cwd(),
        env=environment,
        timeout=max(0, deadline - time.monotonic()),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", choices=("full", "tooling"))
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=os.environ.get("TRADINGDEV_CHECK_SNAPSHOT_TIMEOUT", "5400")
        if os.environ.get("TRADINGDEV_CHECK_SNAPSHOT") == "1"
        else "5400",
    )
    args = parser.parse_args()
    if args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive")
    if os.environ.get("TRADINGDEV_CHECK_SNAPSHOT") == "1":
        run_suite(args.profile, args.timeout_seconds)
    elif git("status", "--porcelain", "--untracked-files=all"):
        print(
            "Uncommitted content: running the requested suite "
            "without reusable evidence.",
            flush=True,
        )
        # A failed explicit run must not expose an earlier successful receipt.
        path = receipt_path(Path.cwd(), args.profile)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with FileLock(str(path.with_suffix(".lock")), timeout=0):
                path.unlink(missing_ok=True)
                run_suite(args.profile, args.timeout_seconds)
        except Timeout as error:
            raise RuntimeError(f"Another {args.profile} check is active") from error
    else:
        check_test_snapshot(
            git("rev-parse", "HEAD^{tree}"),
            args.profile,
            timeout_seconds=args.timeout_seconds,
            reuse=False,
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ProcessCleanupError as error:
        raise SystemExit(report_cleanup_failure(error)) from error
    except subprocess.CalledProcessError as error:
        raise SystemExit(error.returncode) from error
    except (
        OSError,
        RuntimeError,
        EnvironmentCheckError,
        subprocess.TimeoutExpired,
    ) as error:
        print(f"Suite checks failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error
