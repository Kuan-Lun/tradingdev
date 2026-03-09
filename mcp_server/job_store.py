"""Thread-safe and process-safe job state management using file locks.

Uses fcntl.flock() for POSIX file locking (macOS/Linux only) combined
with atomic os.replace() writes to safely share job state between the
MCP server process and worker subprocesses.
"""

from __future__ import annotations

import fcntl
import json
import logging
import math
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

JOBS_DIR = Path(".backtest_jobs")
RESULTS_DIR = JOBS_DIR / "results"
JOBS_FILE = JOBS_DIR / "jobs.json"
LOCK_FILE = JOBS_DIR / "jobs.lock"


def _ensure_dirs() -> None:
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _read_jobs_locked() -> dict[str, Any]:
    _ensure_dirs()
    with open(LOCK_FILE, "a") as lf:
        fcntl.flock(lf, fcntl.LOCK_SH)
        try:
            if JOBS_FILE.exists():
                return dict(json.loads(JOBS_FILE.read_text(encoding="utf-8")))
            return {}
        finally:
            fcntl.flock(lf, fcntl.LOCK_UN)


def _write_jobs_locked(jobs: dict[str, Any]) -> None:
    _ensure_dirs()
    tmp = JOBS_FILE.with_suffix(".tmp")
    with open(LOCK_FILE, "a") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        try:
            tmp.write_text(
                json.dumps(jobs, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            os.replace(tmp, JOBS_FILE)
        finally:
            fcntl.flock(lf, fcntl.LOCK_UN)


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def create_job(
    job_id: str,
    strategy_name: str,
    symbol: str,
    timeframe: str,
    start_date: str,
    end_date: str,
    config_path: str,
) -> dict[str, Any]:
    """Create a new job record and persist it. Returns the record."""
    record: dict[str, Any] = {
        "job_id": job_id,
        "status": "queued",
        "strategy_name": strategy_name,
        "symbol": symbol,
        "timeframe": timeframe,
        "start_date": start_date,
        "end_date": end_date,
        "config_path": config_path,
        "pid": None,
        "start_time": _now_iso(),
        "end_time": None,
        "data_downloaded": False,
        "result_path": str(RESULTS_DIR / f"{job_id}.json"),
        "error": None,
    }
    jobs = _read_jobs_locked()
    jobs[job_id] = record
    _write_jobs_locked(jobs)
    logger.debug("Created job %s", job_id)
    return record


def update_job(job_id: str, **fields: Any) -> None:
    """Update specific fields of an existing job record."""
    jobs = _read_jobs_locked()
    if job_id not in jobs:
        logger.warning("update_job: job %s not found", job_id)
        return
    jobs[job_id].update(fields)
    _write_jobs_locked(jobs)


def get_job(job_id: str) -> dict[str, Any] | None:
    """Return a single job record, or None if not found."""
    jobs = _read_jobs_locked()
    return dict(jobs[job_id]) if job_id in jobs else None


def list_all_jobs() -> list[dict[str, Any]]:
    """Return all job records sorted by start_time descending."""
    jobs = _read_jobs_locked()
    return sorted(
        jobs.values(),
        key=lambda j: j.get("start_time", ""),
        reverse=True,
    )


def save_result(job_id: str, metrics: dict[str, Any]) -> Path:
    """Serialize metrics to JSON and write to results directory.

    Converts all numpy scalar types to Python native types and replaces
    nan/inf with None for JSON compatibility.
    """
    _ensure_dirs()
    safe: dict[str, Any] = {}
    for k, v in metrics.items():
        if isinstance(v, bool):
            safe[k] = bool(v)
        elif isinstance(v, int):
            safe[k] = int(v)
        elif v is None:
            safe[k] = None
        else:
            try:
                f = float(v)
                safe[k] = None if (math.isnan(f) or math.isinf(f)) else f
            except (TypeError, ValueError):
                safe[k] = v

    result_path = RESULTS_DIR / f"{job_id}.json"
    result_path.write_text(
        json.dumps(safe, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.debug("Saved result for job %s → %s", job_id, result_path)
    return result_path


def load_result(result_path: str) -> dict[str, Any] | None:
    """Load a cached result JSON. Returns None if file is missing."""
    path = Path(result_path)
    if not path.exists():
        return None
    return dict(json.loads(path.read_text(encoding="utf-8")))
