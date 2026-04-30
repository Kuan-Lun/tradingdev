"""Typed job/run store facade backed by workspace SQLite metadata."""

from __future__ import annotations

import json
import logging
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from tradingdev.adapters.storage.filesystem import WorkspacePaths, sha256_file
from tradingdev.adapters.storage.sqlite import SQLiteStore

logger = logging.getLogger(__name__)

_WORKSPACE = WorkspacePaths()
_STORE = SQLiteStore(_WORKSPACE)


class JobRecord(BaseModel):
    """Typed job record persisted in SQLite."""

    model_config = ConfigDict(extra="allow")

    job_id: str
    status: str
    strategy_name: str
    symbol: str
    timeframe: str
    start_date: str
    end_date: str
    config_path: str
    job_type: str = "backtest"
    pid: int | None = None
    start_time: str
    end_time: str | None = None
    data_downloaded: bool = False
    result_path: str
    error: str | None = None


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _record_to_dict(record: JobRecord) -> dict[str, Any]:
    return record.model_dump(mode="json")


def create_job(
    job_id: str,
    strategy_name: str,
    symbol: str,
    timeframe: str,
    start_date: str,
    end_date: str,
    config_path: str,
) -> dict[str, Any]:
    """Create a new job record and persist it."""
    result_path = _WORKSPACE.runs / job_id / "result.json"
    record = JobRecord(
        job_id=job_id,
        status="queued",
        strategy_name=strategy_name,
        symbol=symbol,
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date,
        config_path=config_path,
        start_time=_now_iso(),
        result_path=str(result_path),
    )
    payload = _record_to_dict(record)
    _STORE.upsert_job(payload)
    logger.debug("Created job %s", job_id)
    return payload


def update_job(job_id: str, **fields: Any) -> None:
    """Update specific fields of an existing job record."""
    current = _STORE.get_job(job_id)
    if current is None:
        logger.warning("update_job: job %s not found", job_id)
        return
    current.update(fields)
    _STORE.upsert_job(_record_to_dict(JobRecord.model_validate(current)))


def get_job(job_id: str) -> dict[str, Any] | None:
    """Return a single job record, or None if not found."""
    return _STORE.get_job(job_id)


def list_all_jobs() -> list[dict[str, Any]]:
    """Return all job records sorted by start_time descending."""
    return _STORE.list_jobs()


def save_result(job_id: str, metrics: dict[str, Any]) -> Path:
    """Serialize metrics to a run artifact and record run metadata."""
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

    run_dir = _WORKSPACE.runs / job_id
    run_dir.mkdir(parents=True, exist_ok=True)
    result_path = run_dir / "result.json"
    result_path.write_text(
        json.dumps(safe, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    job = get_job(job_id)
    if job is not None:
        config_path = Path(str(job.get("config_path", "")))
        if not config_path.is_absolute():
            config_path = Path.cwd() / config_path
        _STORE.create_run(
            run_id=job_id,
            job_id=job_id,
            strategy_id=str(job.get("strategy_name", "")),
            artifact_dir=run_dir,
            metrics=safe,
            config_hash=sha256_file(config_path) if config_path.exists() else None,
            dataset_id=str(job.get("dataset_id")) if job.get("dataset_id") else None,
        )
        _STORE.create_artifact(
            artifact_id=f"{job_id}:result_json",
            run_id=job_id,
            artifact_type="result_json",
            path=result_path,
            sha256=sha256_file(result_path),
            metadata={"job_id": job_id},
        )
    logger.debug("Saved result for job %s -> %s", job_id, result_path)
    return result_path


def load_result(result_path: str) -> dict[str, Any] | None:
    """Load a cached result JSON. Returns None if file is missing."""
    path = Path(result_path)
    if not path.exists():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    return raw if isinstance(raw, dict) else None


def get_run(run_id: str) -> dict[str, Any] | None:
    """Return a completed run."""
    return _STORE.get_run(run_id)


def list_runs() -> list[dict[str, Any]]:
    """Return completed runs."""
    return _STORE.list_runs()


def list_artifacts(run_id: str | None = None) -> list[dict[str, Any]]:
    """Return stored artifact metadata."""
    return _STORE.list_artifacts(run_id)


def get_artifact(artifact_id: str) -> dict[str, Any] | None:
    """Return stored artifact metadata."""
    return _STORE.get_artifact(artifact_id)
