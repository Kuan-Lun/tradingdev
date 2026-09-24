"""Application service for background jobs."""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal
from uuid import uuid4

from pydantic import ValidationError

from tradingdev.adapters.execution.process_runner import (
    ProcessIdentity,
    ProcessRunner,
    WorkerHandle,
    request_worker_stop,
)
from tradingdev.app.backtest_service import BacktestService
from tradingdev.app.data_service import DataService
from tradingdev.app.job_config import (
    apply_run_overrides,
    bind_strategy_revision,
)
from tradingdev.app.job_store import JobStore, get_default_job_store
from tradingdev.app.strategy_service import (
    StrategyNotExecutableError,
    StrategyService,
)
from tradingdev.domain.backtest.schemas import BacktestRunConfig
from tradingdev.domain.execution import ManifestError
from tradingdev.domain.strategies.loader import StrategyLoader
from tradingdev.shared.utils.config import load_config

if TYPE_CHECKING:
    from tradingdev.domain.strategies.schemas import StrategySpec


class JobService:
    """Create and query background jobs."""

    _ACTIVE_STATUSES = {
        "queued",
        "downloading_data",
        "running_backtest",
        "estimating",
        "pending_confirmation",
        "optimizing",
        "testing_oos",
    }
    _TERMINAL_STATUSES = {"done", "failed", "cancelled", "estimation_timeout"}

    def __init__(
        self,
        *,
        strategy_service: StrategyService | None = None,
        data_service: DataService | None = None,
        strategy_loader: StrategyLoader | None = None,
        job_store: JobStore | None = None,
        process_runner: ProcessRunner | None = None,
        project_root: Path | None = None,
    ) -> None:
        self._job_store = job_store or get_default_job_store()
        self._strategy_service = strategy_service or StrategyService(
            self._job_store.workspace
        )
        self._data_service = data_service or DataService(self._job_store.workspace)
        self._strategy_loader = strategy_loader or StrategyLoader(
            workspace_root=self._job_store.workspace.root
        )
        self._project_root = (project_root or self._default_project_root()).resolve()
        self._process_runner = process_runner or ProcessRunner(
            self._project_root, workspace=self._job_store.workspace
        )

    def start_backtest(
        self,
        *,
        strategy_id: str,
        symbol: str,
        timeframe: str,
        start_date: str,
        end_date: str,
        revision_id: str | None = None,
    ) -> dict[str, Any]:
        """Start a simple backtest job."""
        spec, error = self._resolve_strategy_run_config(strategy_id, revision_id)
        if spec is None:
            return {
                "job_id": "",
                "message": error,
                "data_available": False,
                "code": "strategy_not_executable",
            }
        config_path = Path(spec.config_path)
        raw_config = load_config(config_path)
        run_config = BacktestRunConfig.model_validate(raw_config)
        if run_config.is_walk_forward:
            return {
                "job_id": "",
                "message": (
                    "Config contains validation settings; use start_walk_forward."
                ),
                "data_available": False,
                "code": "invalid_run_mode",
            }
        return self._start_worker(
            strategy_id=strategy_id,
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            config_path=config_path,
            walk_forward=False,
            spec=spec,
        )

    def start_walk_forward(
        self,
        *,
        strategy_id: str,
        symbol: str,
        timeframe: str,
        start_date: str,
        end_date: str,
        revision_id: str | None = None,
    ) -> dict[str, Any]:
        """Start a walk-forward job."""
        spec, error = self._resolve_strategy_run_config(strategy_id, revision_id)
        if spec is None:
            return {
                "job_id": "",
                "message": error,
                "data_available": False,
                "code": "strategy_not_executable",
            }
        config_path, error = self._resolve_walk_forward_config(
            strategy_id=strategy_id,
            config_path=Path(spec.config_path),
            allow_fallback=spec.kind == "bundled",
        )
        if config_path is None:
            return {
                "job_id": "",
                "message": error,
                "data_available": False,
                "code": "invalid_run_mode",
            }
        return self._start_worker(
            strategy_id=strategy_id,
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            config_path=config_path,
            walk_forward=True,
            spec=spec,
        )

    def get_job_status(self, job_id: str) -> dict[str, Any]:
        """Return current job status and completed result payload."""
        job = self._job_store.get_job(job_id)
        if job is None:
            return {
                "status": "not_found",
                "error": f"No job with ID: {job_id}",
                "code": "job_not_found",
            }

        status = str(job["status"])
        if (
            status in self._ACTIVE_STATUSES
            and (status != "queued" or job.get("pid") is not None)
            and not self._is_process_alive(job)
        ):
            status = "failed"
            failure = {
                "status": status,
                "error": "Worker process terminated unexpectedly.",
                "ended_at": datetime.now(UTC).isoformat(),
            }
            self._job_store.update_job(job_id, **failure)
            job.update(failure)

        created_at = datetime.fromisoformat(str(job["created_at"]))
        elapsed = round((datetime.now(UTC) - created_at).total_seconds(), 1)
        response: dict[str, Any] = {
            "status": status,
            "job_type": job.get("job_type", "backtest"),
            "strategy_name": job.get("strategy_name"),
            "revision_id": job.get("revision_id"),
            "manifest_hash": job.get("manifest_hash"),
            "symbol": job.get("symbol"),
            "timeframe": job.get("timeframe"),
            "start_date": job.get("start_date"),
            "end_date": job.get("end_date"),
            "elapsed_seconds": elapsed,
        }

        if status == "done":
            result_path = str(job.get("result_path") or "")
            result = self._job_store.load_result(result_path) if result_path else None
            response["ended_at"] = job.get("ended_at")
            run = self._job_store.get_run(job_id)
            if run is not None:
                response["run_id"] = run["run_id"]
            if job.get("job_type") == "optimization":
                response.update(
                    {
                        "best_params": (result or {}).get("best_params"),
                        "train_metrics": (result or {}).get("train_metrics"),
                        "test_metrics": (result or {}).get("test_metrics"),
                        "optimization_metric": (result or {}).get(
                            "optimization_metric"
                        ),
                        "total_combinations": (result or {}).get("total_combinations"),
                    }
                )
            else:
                response["metrics"] = result
        elif status == "failed":
            response["error"] = job.get("error", "Unknown error")
        elif status == "estimating":
            response["total_combinations"] = job.get("total_combinations")
            response["message"] = "Running trial combination to estimate total time..."
        elif status == "pending_confirmation":
            response.update(
                {
                    "time_per_combo": job.get("time_per_combo"),
                    "total_combinations": job.get("total_combinations"),
                    "estimated_total_seconds": job.get("estimated_total_seconds"),
                    "n_parallel_workers": job.get("n_parallel_workers"),
                    "message": (
                        f"Trial run took {job.get('time_per_combo')}s per combo. "
                        f"Estimated total: {job.get('estimated_total_seconds')}s for "
                        f"{job.get('total_combinations')} combinations using "
                        f"{job.get('n_parallel_workers')} workers. "
                        "Call confirm_optimization(job_id) to proceed."
                    ),
                }
            )
        elif status == "optimizing":
            response["completed"] = job.get("completed", 0)
            response["total_combinations"] = job.get("total_combinations")
            response["estimated_remaining_seconds"] = job.get(
                "estimated_remaining_seconds"
            )
        else:
            response["data_downloaded"] = job.get("data_downloaded", False)
        return response

    def list_jobs(self) -> list[dict[str, Any]]:
        """List job summaries newest first."""
        jobs = self._job_store.list_all_jobs()
        now = datetime.now(UTC)
        summaries = []
        for job in jobs:
            created_at = datetime.fromisoformat(str(job["created_at"]))
            summary: dict[str, Any] = {
                "job_id": job["job_id"],
                "job_type": job.get("job_type", "backtest"),
                "status": job["status"],
                "strategy_name": job.get("strategy_name"),
                "revision_id": job.get("revision_id"),
                "manifest_hash": job.get("manifest_hash"),
                "symbol": job.get("symbol"),
                "timeframe": job.get("timeframe"),
                "start_date": job.get("start_date"),
                "end_date": job.get("end_date"),
                "elapsed_seconds": round((now - created_at).total_seconds(), 1),
                "data_downloaded": job.get("data_downloaded", False),
            }
            if job.get("job_type") == "optimization":
                summary["total_combinations"] = job.get("total_combinations")
                summary["completed"] = job.get("completed", 0)
            summaries.append(summary)
        return summaries

    def confirm_optimization(self, job_id: str) -> dict[str, Any]:
        """Mark an optimization job as confirmed."""
        job = self._job_store.get_job(job_id)
        if job is None:
            return {
                "success": False,
                "error": f"No job with ID: {job_id}",
                "code": "job_not_found",
            }
        status = str(job["status"])
        if status == "estimation_timeout":
            return {
                "success": False,
                "error": "Trial run timed out.",
                "code": "estimation_timeout",
            }
        if status != "pending_confirmation":
            return {
                "success": False,
                "error": f"Job status is '{status}', expected 'pending_confirmation'.",
                "code": "invalid_job_state",
            }
        try:
            self._job_store.load_manifest(job_id)
        except (OSError, ValueError) as exc:
            return {
                "success": False,
                "error": f"Execution specification is invalid: {exc}",
                "code": "execution_manifest_invalid",
            }
        self._job_store.update_job(job_id, confirmed=True)
        return {
            "success": True,
            "message": (
                f"Optimization confirmed. Running {job.get('total_combinations', '?')} "
                f"combinations with {job.get('n_parallel_workers', '?')} workers."
            ),
        }

    def cancel_job(self, job_id: str) -> dict[str, Any]:
        """Cancel a queued or running background job."""
        job = self._job_store.get_job(job_id)
        if job is None:
            return {
                "success": False,
                "error": f"No job with ID: {job_id}",
                "code": "job_not_found",
            }

        status = str(job["status"])
        if status in self._TERMINAL_STATUSES:
            return {
                "success": False,
                "error": f"Job is already terminal: {status}",
                "code": "invalid_job_state",
                "status": status,
            }
        if status not in self._ACTIVE_STATUSES:
            return {
                "success": False,
                "error": f"Job status is not cancellable: {status}",
                "code": "invalid_job_state",
                "status": status,
            }

        process_terminated = False
        handle = WorkerHandle.from_job(job)
        if handle is None:
            if status != "queued" or job.get("pid") is not None:
                return {
                    "success": False,
                    "error": (
                        "Worker control identity is unavailable; "
                        "cannot confirm process cleanup."
                    ),
                    "code": "worker_identity_unavailable",
                    "status": status,
                    "pid": job.get("pid"),
                }
        else:
            try:
                process_terminated = request_worker_stop(
                    self._job_store.workspace.root, handle
                )
            except (OSError, ValueError, RuntimeError) as exc:
                return {
                    "success": False,
                    "error": f"Worker cleanup could not be confirmed: {exc}",
                    "code": "worker_cleanup_failed",
                    "status": status,
                    "pid": handle.pid,
                }

        self._job_store.update_job(
            job_id,
            status="cancelled",
            error="Cancelled by user.",
            ended_at=datetime.now(UTC).isoformat(),
        )
        return {
            "success": True,
            "job_id": job_id,
            "status": "cancelled",
            "process_terminated": process_terminated,
        }

    def _start_worker(
        self,
        *,
        strategy_id: str,
        symbol: str,
        timeframe: str,
        start_date: str,
        end_date: str,
        config_path: Path,
        walk_forward: bool,
        spec: StrategySpec,
    ) -> dict[str, Any]:
        raw_config = load_config(config_path)
        bind_strategy_revision(raw_config, spec)
        effective_config = apply_run_overrides(
            raw_config,
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
        )
        kind: Literal["backtest", "walk_forward"] = (
            "walk_forward" if walk_forward else "backtest"
        )
        try:
            manifest = BacktestService(
                data_service=self._data_service,
                strategy_gate=self._strategy_service,
                strategy_loader=self._strategy_loader,
            ).prepare_execution(effective_config, kind=kind)
        except (ManifestError, ValidationError) as exc:
            return {
                "job_id": "",
                "message": str(exc),
                "data_available": False,
                "code": "invalid_execution_request",
            }

        data_available = self._data_service.data_available(
            symbol,
            timeframe,
            start_date,
            end_date,
        )
        job_id = uuid4().hex[:12]
        self._job_store.create_job(
            job_id=job_id,
            strategy_name=strategy_id,
            revision_id=spec.revision_id,
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            job_type=kind,
            manifest=manifest,
        )
        self._job_store.update_job(
            job_id,
            original_config_path=str(config_path),
        )
        try:
            identity = self._process_runner.spawn_module(
                "tradingdev.mcp.workers.backtest",
                job_id,
            )
        except BaseException as exc:
            # Interruptions must not leave a job queued after startup cleanup.
            self._job_store.update_job(
                job_id,
                status="failed",
                error=f"Worker failed to start: {type(exc).__name__}: {exc}",
            )
            raise
        self._job_store.update_job(job_id, **identity.job_fields())
        data_msg = (
            "Data already cached locally."
            if data_available
            else "Data not fully cached; worker will download it automatically."
        )
        return {
            "job_id": job_id,
            "revision_id": spec.revision_id,
            "manifest_hash": manifest.manifest_hash,
            "message": f"Job started. Job ID: {job_id}. {data_msg}",
            "data_available": data_available,
        }

    def _resolve_walk_forward_config(
        self,
        *,
        strategy_id: str,
        config_path: Path,
        allow_fallback: bool,
    ) -> tuple[Path | None, str]:
        raw_config = load_config(config_path)
        run_config = BacktestRunConfig.model_validate(raw_config)
        if run_config.is_walk_forward:
            return config_path, ""

        fallback = config_path.with_name("walkforward_config.yaml")
        if allow_fallback and fallback.exists():
            fallback_raw = load_config(fallback)
            fallback_run_config = BacktestRunConfig.model_validate(fallback_raw)
            fallback_strategy = fallback_raw.get("strategy", {})
            fallback_id = (
                fallback_strategy.get("id")
                if isinstance(fallback_strategy, dict)
                else None
            )
            if fallback_id == strategy_id and fallback_run_config.is_walk_forward:
                return fallback, ""
        return None, "Config has no validation section for walk-forward."

    def _resolve_strategy_run_config(
        self, strategy_id: str, revision_id: str | None
    ) -> tuple[StrategySpec | None, str]:
        try:
            spec = self._strategy_service.resolve_executable(strategy_id, revision_id)
        except StrategyNotExecutableError as exc:
            return None, str(exc)
        path = Path(spec.config_path)
        return (spec, "") if path.exists() else (None, f"Config not found: {path}")

    @staticmethod
    def _process_identity(job: dict[str, Any]) -> ProcessIdentity | None:
        return ProcessIdentity.from_values(
            job.get("pid"), job.get("process_create_time")
        )

    def _is_process_alive(self, job: dict[str, Any]) -> bool:
        identity = self._process_identity(job)
        return identity is not None and identity.get_process() is not None

    def _default_project_root(self) -> Path:
        configured = os.environ.get("TRADINGDEV_PROJECT_ROOT")
        if configured:
            return Path(configured).expanduser().resolve()
        return Path.cwd().resolve()
