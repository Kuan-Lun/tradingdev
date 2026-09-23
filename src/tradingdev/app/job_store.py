"""Typed job/run store facade backed by workspace SQLite metadata."""

from __future__ import annotations

import json
import logging
import pickle
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict

from tradingdev.adapters.storage.filesystem import (
    WorkspacePaths,
    sha256_file,
    sha256_text,
)
from tradingdev.adapters.storage.sqlite import SQLiteStore, get_sqlite_store
from tradingdev.app.run_lineage import (
    extract_random_seed,
    load_config_payload,
    read_strategy_snapshot,
)
from tradingdev.app.strategy_service import StrategyNotExecutableError
from tradingdev.shared.utils.json_values import normalize_json_object

logger = logging.getLogger(__name__)


class JobRecord(BaseModel):
    """Typed job record persisted in SQLite."""

    model_config = ConfigDict(extra="allow")

    job_id: str
    status: str
    strategy_name: str | None = None
    revision_id: str | None = None
    symbol: str | None = None
    timeframe: str | None = None
    start_date: str | None = None
    end_date: str | None = None
    config_path: str | None = None
    job_type: str = "backtest"
    pid: int | None = None
    process_create_time: float | None = None
    worker_control_id: str | None = None
    created_at: str
    started_at: str | None = None
    ended_at: str | None = None
    data_downloaded: bool = False
    result_path: str | None = None
    error: str | None = None


class JobStore:
    """Application-level facade for job, run, and artifact metadata."""

    def __init__(
        self,
        *,
        workspace: WorkspacePaths | None = None,
        store: SQLiteStore | None = None,
    ) -> None:
        if workspace is not None:
            self._workspace = workspace
        elif store is not None:
            self._workspace = WorkspacePaths(store.db_path.parent)
        else:
            self._workspace = WorkspacePaths()
        self._workspace.ensure()
        self._store = store or get_sqlite_store(self._workspace)

    @property
    def workspace(self) -> WorkspacePaths:
        """Return the runtime workspace used by this facade."""
        return self._workspace

    @property
    def store(self) -> SQLiteStore:
        """Return the backing SQLite adapter."""
        return self._store

    def create_job(
        self,
        *,
        job_id: str,
        strategy_name: str | None = None,
        revision_id: str | None = None,
        symbol: str | None = None,
        timeframe: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        config_path: str | None = None,
        job_type: str = "backtest",
        extra_payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create a new job record and persist it."""
        result_path = self._workspace.runs / job_id / "result.json"
        record = JobRecord(
            job_id=job_id,
            status="queued",
            job_type=job_type,
            strategy_name=strategy_name,
            revision_id=revision_id,
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            config_path=config_path,
            created_at=_now_iso(),
            result_path=str(result_path),
        )
        record_payload = _record_to_dict(record)
        if extra_payload is not None:
            record_payload.update(extra_payload)
        self._store.upsert_job(record_payload)
        logger.debug("Created job %s", job_id)
        return record_payload

    def update_job(self, job_id: str, **fields: Any) -> None:
        """Update specific fields of an existing job record."""
        current = self._store.get_job(job_id)
        if current is None:
            logger.warning("update_job: job %s not found", job_id)
            return
        status = fields.get("status")
        if (
            status
            in {
                "downloading_data",
                "running_backtest",
                "estimating",
                "optimizing",
                "testing_oos",
            }
            and current.get("started_at") is None
            and "started_at" not in fields
        ):
            fields["started_at"] = _now_iso()
        if (
            status in {"done", "failed", "cancelled", "estimation_timeout"}
            and "ended_at" not in fields
        ):
            fields["ended_at"] = _now_iso()
        current.update(fields)
        self._store.upsert_job(_record_to_dict(JobRecord.model_validate(current)))

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        """Return a single job record, or None if not found."""
        return self._store.get_job(job_id)

    def list_all_jobs(self) -> list[dict[str, Any]]:
        """Return all job records sorted by creation time descending."""
        return self._store.list_jobs()

    def save_result(
        self,
        job_id: str,
        metrics: dict[str, Any],
        *,
        pipeline: Any | None = None,
        config_snapshot: dict[str, Any] | None = None,
    ) -> Path:
        """Serialize metrics to a run artifact and record run metadata."""
        safe = normalize_json_object(metrics)

        job = self.get_job(job_id)
        config_path = _resolve_optional_path(job.get("config_path")) if job else None
        config_payload = (
            pipeline.config_snapshot
            if pipeline is not None
            else config_snapshot
            if config_snapshot is not None
            else load_config_payload(config_path)
            if config_path is not None
            else None
        )
        source = read_strategy_snapshot(
            config_payload,
            self._workspace,
            strategy_id=str((job or {}).get("strategy_name") or ""),
        )
        if job is not None and job.get("revision_id") != source.revision_id:
            msg = "Job and result configuration refer to different revisions"
            raise StrategyNotExecutableError(msg)
        config_content = (
            yaml.safe_dump(config_payload, sort_keys=False, allow_unicode=True)
            if config_payload is not None
            else None
        )
        config_hash = (
            sha256_text(config_content) if config_content is not None else None
        )
        run_dir = self._workspace.runs / job_id
        run_dir.mkdir(parents=True, exist_ok=True)
        result_path = run_dir / "result.json"
        result_path.write_text(
            json.dumps(safe, indent=2, ensure_ascii=False, allow_nan=False),
            encoding="utf-8",
        )

        if job is not None:
            strategy_source = source.path
            source_hash = source.source_hash
            random_seed = extract_random_seed(config_payload)
            dataset_fingerprint = self._dataset_fingerprint(job)
            # Current execution creates one run per job, so run_id is job_id.
            # Revisit this before supporting multi-run jobs such as fold-level
            # optimization artifacts.
            self._store.create_run(
                run_id=job_id,
                job_id=job_id,
                strategy_id=str(job.get("strategy_name", "")),
                revision_id=source.revision_id,
                artifact_dir=run_dir,
                metrics=safe,
                config_hash=config_hash,
                source_hash=source_hash,
                random_seed=random_seed,
                dataset_id=dataset_fingerprint["dataset_id"],
            )
            self._store.create_artifact(
                artifact_id=f"{job_id}:result_json",
                run_id=job_id,
                artifact_type="result_json",
                path=result_path,
                sha256=sha256_file(result_path),
                metadata={"job_id": job_id},
            )
            if config_content is not None:
                snapshot_path = run_dir / "config.yaml"
                snapshot_path.write_bytes(config_content.encode("utf-8"))
                self._store.create_artifact(
                    artifact_id=f"{job_id}:config_snapshot",
                    run_id=job_id,
                    artifact_type="config_snapshot",
                    path=snapshot_path,
                    sha256=sha256_file(snapshot_path),
                    metadata={
                        "job_id": job_id,
                        "source_path": str(config_path),
                        "config_hash": config_hash,
                    },
                )
                if strategy_source is not None and source.content is not None:
                    strategy_snapshot = run_dir / "strategy.py"
                    strategy_snapshot.write_bytes(source.content)
                    self._store.create_artifact(
                        artifact_id=f"{job_id}:strategy_source",
                        run_id=job_id,
                        artifact_type="strategy_source",
                        path=strategy_snapshot,
                        sha256=sha256_file(strategy_snapshot),
                        metadata={
                            "job_id": job_id,
                            "source_path": str(strategy_source),
                            "source_hash": source_hash,
                            "revision_id": source.revision_id,
                        },
                    )
            fingerprint_path = run_dir / "dataset_fingerprint.json"
            fingerprint_path.write_text(
                json.dumps(dataset_fingerprint, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            self._store.create_artifact(
                artifact_id=f"{job_id}:dataset_fingerprint",
                run_id=job_id,
                artifact_type="dataset_fingerprint",
                path=fingerprint_path,
                sha256=sha256_file(fingerprint_path),
                metadata=dataset_fingerprint,
            )
            if pipeline is not None:
                pipeline_path = run_dir / "pipeline_result.pkl"
                pipeline_path.write_bytes(pickle.dumps(pipeline))
                self._store.create_artifact(
                    artifact_id=f"{job_id}:pipeline_result",
                    run_id=job_id,
                    artifact_type="pipeline_result",
                    path=pipeline_path,
                    sha256=sha256_file(pipeline_path),
                    metadata={"job_id": job_id, "format": "pickle"},
                )
        logger.debug("Saved result for job %s -> %s", job_id, result_path)
        return result_path

    def load_result(self, result_path: str) -> dict[str, Any] | None:
        """Load a cached result JSON. Returns None if file is missing."""
        path = Path(result_path)
        if not path.exists():
            return None
        raw = json.loads(path.read_text(encoding="utf-8"))
        return normalize_json_object(raw) if isinstance(raw, dict) else None

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        """Return a completed run."""
        return self._store.get_run(run_id)

    def list_runs(self) -> list[dict[str, Any]]:
        """Return completed runs."""
        return self._store.list_runs()

    def list_artifacts(self, run_id: str | None = None) -> list[dict[str, Any]]:
        """Return stored artifact metadata."""
        return self._store.list_artifacts(run_id)

    def get_artifact(self, artifact_id: str) -> dict[str, Any] | None:
        """Return stored artifact metadata."""
        return self._store.get_artifact(artifact_id)

    def _dataset_fingerprint(self, job: dict[str, Any]) -> dict[str, str]:
        dataset_id = str(job.get("dataset_id") or "")
        payload = {
            "dataset_id": dataset_id,
            "symbol": str(job.get("symbol") or ""),
            "timeframe": str(job.get("timeframe") or ""),
            "start_date": str(job.get("start_date") or ""),
            "end_date": str(job.get("end_date") or ""),
        }
        if not dataset_id:
            dataset_id = sha256_text(json.dumps(payload, sort_keys=True))
            payload["dataset_id"] = dataset_id
        payload["fingerprint"] = sha256_text(json.dumps(payload, sort_keys=True))
        return payload


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _record_to_dict(record: JobRecord) -> dict[str, Any]:
    return record.model_dump(mode="json")


def _resolve_optional_path(value: object) -> Path | None:
    if not value:
        return None
    path = Path(str(value))
    if not path.is_absolute():
        path = Path.cwd() / path
    return path


def get_default_job_store() -> JobStore:
    """Return a default job store without import-time workspace side effects."""
    return JobStore()


def create_job(
    job_id: str,
    strategy_name: str | None = None,
    symbol: str | None = None,
    timeframe: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    config_path: str | None = None,
    job_type: str = "backtest",
    extra_payload: dict[str, Any] | None = None,
    revision_id: str | None = None,
) -> dict[str, Any]:
    """Create a new job record in the default store."""
    return get_default_job_store().create_job(
        job_id=job_id,
        strategy_name=strategy_name,
        revision_id=revision_id,
        symbol=symbol,
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date,
        config_path=config_path,
        job_type=job_type,
        extra_payload=extra_payload,
    )


def update_job(job_id: str, **fields: Any) -> None:
    """Update a job in the default store."""
    get_default_job_store().update_job(job_id, **fields)


def get_job(job_id: str) -> dict[str, Any] | None:
    """Return a job from the default store."""
    return get_default_job_store().get_job(job_id)


def list_all_jobs() -> list[dict[str, Any]]:
    """Return jobs from the default store."""
    return get_default_job_store().list_all_jobs()


def save_result(
    job_id: str,
    metrics: dict[str, Any],
    *,
    pipeline: Any | None = None,
    config_snapshot: dict[str, Any] | None = None,
) -> Path:
    """Persist a result in the default store."""
    return get_default_job_store().save_result(
        job_id,
        metrics,
        pipeline=pipeline,
        config_snapshot=config_snapshot,
    )


def load_result(result_path: str) -> dict[str, Any] | None:
    """Load a result from disk."""
    return get_default_job_store().load_result(result_path)


def get_run(run_id: str) -> dict[str, Any] | None:
    """Return a completed run from the default store."""
    return get_default_job_store().get_run(run_id)


def list_runs() -> list[dict[str, Any]]:
    """Return completed runs from the default store."""
    return get_default_job_store().list_runs()


def list_artifacts(run_id: str | None = None) -> list[dict[str, Any]]:
    """Return artifacts from the default store."""
    return get_default_job_store().list_artifacts(run_id)


def get_artifact(artifact_id: str) -> dict[str, Any] | None:
    """Return one artifact from the default store."""
    return get_default_job_store().get_artifact(artifact_id)
