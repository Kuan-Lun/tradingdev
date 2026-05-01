"""Application service for background jobs."""

from __future__ import annotations

import os
import signal
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import yaml

from tradingdev.adapters.execution.process_runner import ProcessRunner
from tradingdev.app.data_service import DataService
from tradingdev.app.job_store import JobStore, get_default_job_store
from tradingdev.app.strategy_service import StrategyService
from tradingdev.domain.backtest.schemas import BacktestRunConfig
from tradingdev.shared.utils.config import load_config


class JobService:
    """Create and query background jobs."""

    _RUNNABLE_STATUSES = {"runnable", "promoted"}
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
        job_store: JobStore | None = None,
        process_runner: ProcessRunner | None = None,
        project_root: Path | None = None,
    ) -> None:
        self._strategy_service = strategy_service or StrategyService()
        self._data_service = data_service or DataService()
        self._job_store = job_store or get_default_job_store()
        self._project_root = (project_root or self._default_project_root()).resolve()
        self._process_runner = process_runner or ProcessRunner(self._project_root)

    def start_backtest(
        self,
        *,
        strategy_id: str,
        symbol: str,
        timeframe: str,
        start_date: str,
        end_date: str,
    ) -> dict[str, Any]:
        """Start a simple backtest job."""
        config_path, error = self._resolve_strategy_run_config(strategy_id)
        if config_path is None:
            return {"job_id": "", "message": error, "data_available": False}
        raw_config = load_config(config_path)
        run_config = BacktestRunConfig.model_validate(raw_config)
        if run_config.is_walk_forward:
            return {
                "job_id": "",
                "message": (
                    "Config contains validation settings; use start_walk_forward."
                ),
                "data_available": False,
            }
        return self._start_worker(
            strategy_id=strategy_id,
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            config_path=config_path,
            walk_forward=False,
        )

    def start_walk_forward(
        self,
        *,
        strategy_id: str,
        symbol: str,
        timeframe: str,
        start_date: str,
        end_date: str,
    ) -> dict[str, Any]:
        """Start a walk-forward job."""
        config_path, error = self._resolve_strategy_run_config(strategy_id)
        if config_path is None:
            return {"job_id": "", "message": error, "data_available": False}
        config_path, error = self._resolve_walk_forward_config(
            strategy_id=strategy_id,
            config_path=config_path,
        )
        if config_path is None:
            return {
                "job_id": "",
                "message": error,
                "data_available": False,
            }
        return self._start_worker(
            strategy_id=strategy_id,
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            config_path=config_path,
            walk_forward=True,
        )

    def get_job_status(self, job_id: str) -> dict[str, Any]:
        """Return current job status and completed result payload."""
        job = self._job_store.get_job(job_id)
        if job is None:
            return {"status": "not_found", "error": f"No job with ID: {job_id}"}

        status = str(job["status"])
        if status in {
            "downloading_data",
            "running_backtest",
            "estimating",
            "optimizing",
            "testing_oos",
        } and not self._is_pid_alive(job.get("pid")):
            status = "failed"
            self._job_store.update_job(
                job_id,
                status=status,
                error="Worker process terminated unexpectedly.",
                ended_at=datetime.now(UTC).isoformat(),
            )

        created_at = datetime.fromisoformat(str(job["created_at"]))
        elapsed = round((datetime.now(UTC) - created_at).total_seconds(), 1)
        response: dict[str, Any] = {
            "status": status,
            "job_type": job.get("job_type", "backtest"),
            "strategy_name": job.get("strategy_name"),
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
            return {"success": False, "error": f"No job with ID: {job_id}"}
        status = str(job["status"])
        if status == "estimation_timeout":
            return {"success": False, "error": "Trial run timed out."}
        if status != "pending_confirmation":
            return {
                "success": False,
                "error": f"Job status is '{status}', expected 'pending_confirmation'.",
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
            return {"success": False, "error": f"No job with ID: {job_id}"}

        status = str(job["status"])
        if status in self._TERMINAL_STATUSES:
            return {
                "success": False,
                "error": f"Job is already terminal: {status}",
                "status": status,
            }
        if status not in self._ACTIVE_STATUSES:
            return {
                "success": False,
                "error": f"Job status is not cancellable: {status}",
                "status": status,
            }

        process_terminated = False
        pid = job.get("pid")
        if isinstance(pid, int) and self._is_pid_alive(pid):
            process_terminated, error = self._terminate_pid(pid)
            if error is not None:
                return {
                    "success": False,
                    "error": error,
                    "status": status,
                    "pid": pid,
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
    ) -> dict[str, Any]:
        raw_config = load_config(config_path)
        effective_config = self._apply_run_overrides(
            raw_config,
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
        )
        BacktestRunConfig.model_validate(effective_config)

        data_available = self._data_service.data_available(
            symbol,
            timeframe,
            start_date,
            end_date,
        )
        job_id = uuid4().hex[:12]
        effective_config_path = self._write_job_config(job_id, effective_config)
        self._job_store.create_job(
            job_id=job_id,
            strategy_name=strategy_id,
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            config_path=str(effective_config_path),
        )
        self._job_store.update_job(
            job_id,
            job_type="walk_forward" if walk_forward else "backtest",
            original_config_path=str(config_path),
        )
        args = [job_id, str(effective_config_path)]
        if walk_forward:
            args.append("--walk-forward")
        pid = self._process_runner.spawn_module(
            "tradingdev.mcp.workers.backtest",
            *args,
        )
        self._job_store.update_job(job_id, pid=pid)
        data_msg = (
            "Data already cached locally."
            if data_available
            else "Data not fully cached; worker will download it automatically."
        )
        return {
            "job_id": job_id,
            "message": f"Job started. Job ID: {job_id}. {data_msg}",
            "data_available": data_available,
        }

    def _apply_run_overrides(
        self,
        raw_config: dict[str, Any],
        *,
        symbol: str,
        timeframe: str,
        start_date: str,
        end_date: str,
    ) -> dict[str, Any]:
        effective_config = deepcopy(raw_config)
        backtest = effective_config.get("backtest")
        if not isinstance(backtest, dict):
            msg = "backtest config must be a mapping"
            raise ValueError(msg)
        backtest.update(
            {
                "symbol": symbol,
                "timeframe": timeframe,
                "start_date": start_date,
                "end_date": end_date,
            }
        )

        data = effective_config.get("data")
        if isinstance(data, dict):
            requirements = data.get("requirements")
            if isinstance(requirements, dict):
                market = requirements.get("market")
                if isinstance(market, dict):
                    market["symbol"] = symbol
                    market["timeframe"] = timeframe
        return effective_config

    def _resolve_walk_forward_config(
        self,
        *,
        strategy_id: str,
        config_path: Path,
    ) -> tuple[Path | None, str]:
        raw_config = load_config(config_path)
        run_config = BacktestRunConfig.model_validate(raw_config)
        if run_config.is_walk_forward:
            return config_path, ""

        fallback = config_path.with_name("walkforward_config.yaml")
        if fallback.exists():
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

    def _write_job_config(self, job_id: str, config: dict[str, Any]) -> Path:
        run_dir = self._job_store.workspace.runs / job_id
        run_dir.mkdir(parents=True, exist_ok=True)
        path = run_dir / "config.yaml"
        path.write_text(
            yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        return path

    def _resolve_strategy_run_config(self, strategy_id: str) -> tuple[Path | None, str]:
        strategy = self._strategy_service.get_strategy(strategy_id)
        if not strategy.get("success"):
            return None, str(strategy.get("error", "Strategy not found"))
        metadata = strategy.get("metadata", {})
        if not isinstance(metadata, dict):
            return None, "Strategy metadata is invalid"
        status = str(metadata.get("status", "draft"))
        if status not in self._RUNNABLE_STATUSES:
            return None, (
                "Strategy must be runnable or promoted before execution. "
                f"Current status: {status}"
            )
        config_path = metadata.get("config_path")
        if not config_path:
            return None, "Strategy metadata has no config_path"
        path = Path(str(config_path))
        return (path, "") if path.exists() else (None, f"Config not found: {path}")

    def _is_pid_alive(self, pid: object) -> bool:
        if not isinstance(pid, int):
            return False
        try:
            os.kill(pid, 0)
            return True
        except (ProcessLookupError, PermissionError):
            return False

    def _terminate_pid(self, pid: int) -> tuple[bool, str | None]:
        try:
            os.killpg(pid, signal.SIGTERM)
            return True, None
        except ProcessLookupError:
            return False, None
        except PermissionError:
            return False, f"Permission denied when terminating process group {pid}"
        except OSError:
            try:
                os.kill(pid, signal.SIGTERM)
                return True, None
            except ProcessLookupError:
                return False, None
            except PermissionError:
                return False, f"Permission denied when terminating process {pid}"

    def _default_project_root(self) -> Path:
        configured = os.environ.get("TRADINGDEV_PROJECT_ROOT")
        if configured:
            return Path(configured).expanduser().resolve()
        return Path.cwd().resolve()
