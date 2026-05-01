"""Application service for parameter optimization jobs."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any
from uuid import uuid4

from tradingdev.adapters.execution.process_runner import ProcessRunner
from tradingdev.app.job_store import JobStore, get_default_job_store
from tradingdev.app.strategy_service import StrategyService


class OptimizationService:
    """Create background optimization jobs."""

    _VALID_METRICS = frozenset(
        {
            "total_return",
            "total_pnl",
            "annual_return",
            "sharpe_ratio",
            "max_drawdown",
            "win_rate",
            "profit_factor",
        }
    )

    def __init__(
        self,
        *,
        strategy_service: StrategyService | None = None,
        job_store: JobStore | None = None,
        process_runner: ProcessRunner | None = None,
        project_root: Path | None = None,
    ) -> None:
        self._strategy_service = strategy_service or StrategyService()
        self._job_store = job_store or get_default_job_store()
        self._project_root = (project_root or Path.cwd()).resolve()
        self._process_runner = process_runner or ProcessRunner(self._project_root)

    def start_optimization(
        self,
        *,
        strategy_id: str,
        symbol: str,
        timeframe: str,
        param_ranges: dict[str, list[Any]],
        optimization_metric: str,
        train_start: str,
        train_end: str,
        test_start: str,
        test_end: str,
    ) -> dict[str, Any]:
        """Start a parameter optimization worker."""
        config_path, error = self._resolve_strategy_config(strategy_id)
        if config_path is None:
            return {"job_id": "", "message": error, "total_combinations": 0}
        validation_error = self._validate_request(
            param_ranges,
            optimization_metric,
            train_start,
            train_end,
            test_start,
            test_end,
        )
        if validation_error:
            return {"job_id": "", "message": validation_error, "total_combinations": 0}

        total_combinations = 1
        for values in param_ranges.values():
            total_combinations *= len(values)

        job_id = uuid4().hex[:12]
        self._job_store.create_job(
            job_id=job_id,
            strategy_name=strategy_id,
            symbol=symbol,
            timeframe=timeframe,
            start_date=train_start,
            end_date=test_end,
            config_path=str(config_path),
        )
        self._job_store.update_job(
            job_id,
            job_type="optimization",
            param_ranges=param_ranges,
            optimization_metric=optimization_metric,
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
            total_combinations=total_combinations,
        )
        pid = self._process_runner.spawn_module(
            "tradingdev.mcp.workers.optimization",
            job_id,
        )
        self._job_store.update_job(job_id, pid=pid)
        return {
            "job_id": job_id,
            "message": (
                f"Optimization started. {total_combinations} parameter combinations. "
                "A trial run will estimate total time; use get_job_status() to check."
            ),
            "total_combinations": total_combinations,
        }

    def _resolve_strategy_config(self, strategy_id: str) -> tuple[Path | None, str]:
        strategy = self._strategy_service.get_strategy(strategy_id)
        if not strategy.get("success"):
            return None, str(strategy.get("error", "Strategy not found"))
        metadata = strategy.get("metadata", {})
        if not isinstance(metadata, dict):
            return None, "Strategy metadata is invalid"
        if str(metadata.get("status", "draft")) not in {"runnable", "promoted"}:
            return None, "Strategy must be runnable or promoted before optimization."
        path = Path(str(metadata.get("config_path", "")))
        return (path, "") if path.exists() else (None, f"Config not found: {path}")

    def _validate_request(
        self,
        param_ranges: dict[str, list[Any]],
        metric: str,
        train_start: str,
        train_end: str,
        test_start: str,
        test_end: str,
    ) -> str:
        if not param_ranges:
            return "param_ranges must not be empty."
        for name, values in param_ranges.items():
            if not isinstance(values, list) or not values:
                return f"param_ranges['{name}'] must be a non-empty list."
        if metric not in self._VALID_METRICS:
            return (
                f"Invalid metric '{metric}'. Choose from: {sorted(self._VALID_METRICS)}"
            )
        try:
            ts = date.fromisoformat(train_start)
            te = date.fromisoformat(train_end)
            vs = date.fromisoformat(test_start)
            ve = date.fromisoformat(test_end)
        except ValueError as exc:
            return f"Invalid date format: {exc}"
        if not (ts < te <= vs < ve):
            return (
                "Dates must satisfy: train_start < train_end <= test_start < test_end. "
                f"Got: {train_start} < {train_end} <= {test_start} < {test_end}"
            )
        return ""
