"""Application service for parameter optimization jobs."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from tradingdev.adapters.execution.process_runner import ProcessRunner
from tradingdev.app.job_config import (
    apply_run_overrides,
    bind_strategy_revision,
    write_job_config,
)
from tradingdev.app.job_store import JobStore, get_default_job_store
from tradingdev.app.strategy_service import (
    StrategyNotExecutableError,
    StrategyService,
)
from tradingdev.domain.backtest.schemas import BacktestRunConfig
from tradingdev.shared.utils.config import load_config

if TYPE_CHECKING:
    from tradingdev.domain.strategies.schemas import StrategySpec


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
        self._job_store = job_store or get_default_job_store()
        self._strategy_service = strategy_service or StrategyService(
            self._job_store.workspace
        )
        self._process_runner = process_runner or ProcessRunner(
            project_root, workspace=self._job_store.workspace
        )

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
        revision_id: str | None = None,
    ) -> dict[str, Any]:
        """Start a parameter optimization worker."""
        spec, error = self._resolve_strategy_config(strategy_id, revision_id)
        if spec is None:
            return {
                "job_id": "",
                "message": error,
                "total_combinations": 0,
                "code": "strategy_not_executable",
            }
        validation_error = self._validate_request(
            param_ranges,
            optimization_metric,
            train_start,
            train_end,
            test_start,
            test_end,
        )
        if validation_error:
            return {
                "job_id": "",
                "message": validation_error,
                "total_combinations": 0,
                "code": "invalid_optimization_request",
            }

        total_combinations = 1
        for values in param_ranges.values():
            total_combinations *= len(values)

        config_path = Path(spec.config_path)
        raw_config = load_config(config_path)
        bind_strategy_revision(raw_config, spec)
        effective_config = apply_run_overrides(
            raw_config,
            symbol=symbol,
            timeframe=timeframe,
            start_date=train_start,
            # Optimization bounds are calendar days, including the final day.
            end_date=f"{test_end}T23:59:59.999999",
        )
        BacktestRunConfig.model_validate(effective_config)
        job_id = uuid4().hex[:12]
        effective_path = write_job_config(
            self._job_store.workspace.runs / job_id, effective_config
        )
        self._job_store.create_job(
            job_id=job_id,
            strategy_name=strategy_id,
            revision_id=spec.revision_id,
            symbol=symbol,
            timeframe=timeframe,
            start_date=train_start,
            end_date=test_end,
            config_path=str(effective_path),
        )
        self._job_store.update_job(
            job_id,
            job_type="optimization",
            original_config_path=str(config_path),
            param_ranges=param_ranges,
            optimization_metric=optimization_metric,
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
            total_combinations=total_combinations,
        )
        try:
            identity = self._process_runner.spawn_module(
                "tradingdev.mcp.workers.optimization",
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
        return {
            "job_id": job_id,
            "revision_id": spec.revision_id,
            "message": (
                f"Optimization started. {total_combinations} parameter combinations. "
                "A trial run will estimate total time; use get_job_status() to check."
            ),
            "total_combinations": total_combinations,
        }

    def _resolve_strategy_config(
        self, strategy_id: str, revision_id: str | None
    ) -> tuple[StrategySpec | None, str]:
        try:
            spec = self._strategy_service.resolve_executable(strategy_id, revision_id)
        except StrategyNotExecutableError as exc:
            return None, str(exc)
        path = Path(spec.config_path)
        return (spec, "") if path.exists() else (None, f"Config not found: {path}")

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
        if not (ts < te < vs < ve):
            return (
                "Dates must satisfy: train_start < train_end < test_start < test_end "
                "(inclusive calendar days, without overlap). "
                f"Got: {train_start}, {train_end}, {test_start}, {test_end}"
            )
        return ""
