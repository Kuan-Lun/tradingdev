"""Application service for parameter optimization jobs."""

from __future__ import annotations

from itertools import product
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from pydantic import ValidationError

from tradingdev.adapters.execution.process_runner import ProcessRunner
from tradingdev.app.backtest_service import BacktestService
from tradingdev.app.data_service import DataService
from tradingdev.app.job_config import apply_run_overrides
from tradingdev.app.job_store import JobStore, get_default_job_store
from tradingdev.app.strategy_service import (
    StrategyNotExecutableError,
    StrategyService,
)
from tradingdev.domain.execution import ManifestError, OptimizationSpec
from tradingdev.domain.strategies.loader import StrategyLoader
from tradingdev.shared.utils.config import load_config

if TYPE_CHECKING:
    from tradingdev.domain.strategies.schemas import StrategySpec


class OptimizationService:
    """Create background optimization jobs."""

    def __init__(
        self,
        *,
        strategy_service: StrategyService | None = None,
        job_store: JobStore | None = None,
        process_runner: ProcessRunner | None = None,
        strategy_loader: StrategyLoader | None = None,
        project_root: Path | None = None,
    ) -> None:
        self._job_store = job_store or get_default_job_store()
        self._strategy_service = strategy_service or StrategyService(
            self._job_store.workspace
        )
        self._process_runner = process_runner or ProcessRunner(
            project_root, workspace=self._job_store.workspace
        )
        self._strategy_loader = strategy_loader or StrategyLoader(
            workspace_root=self._job_store.workspace.root
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
        try:
            optimization = OptimizationSpec.model_validate(
                {
                    "param_ranges": param_ranges,
                    "optimization_metric": optimization_metric,
                    "train_start": train_start,
                    "train_end": train_end,
                    "test_start": test_start,
                    "test_end": test_end,
                }
            )
        except ValidationError as exc:
            return {
                "job_id": "",
                "message": str(exc),
                "total_combinations": 0,
                "code": "invalid_optimization_request",
            }

        config_path = Path(spec.config_path)
        raw_config = load_config(config_path)
        if raw_config.get("validation") is not None:
            return {
                "job_id": "",
                "message": (
                    "Optimization config must not contain validation settings; "
                    "use a config without walk-forward validation."
                ),
                "total_combinations": 0,
                "code": "invalid_optimization_request",
            }
        effective_config = apply_run_overrides(
            raw_config,
            symbol=symbol,
            timeframe=timeframe,
            start_date=train_start,
            # Optimization bounds are calendar days, including the final day.
            end_date=f"{test_end}T23:59:59.999999",
        )
        try:
            manifest = BacktestService(
                strategy_gate=self._strategy_service,
                strategy_loader=self._strategy_loader,
                data_service=DataService(self._job_store.workspace),
            ).prepare_execution(
                effective_config, kind="optimization", optimization=optimization
            )
            strategy_config = manifest.config_copy()["strategy"]
            names = list(optimization.param_ranges)
            candidates = (optimization.param_ranges[name] for name in names)
            combinations = (
                dict(zip(names, combination, strict=True))
                for combination in product(*candidates)
            )
            try:
                self._strategy_loader.validate_parameter_grid(
                    strategy_config, manifest.strategy_execution, combinations
                )
            except Exception as exc:
                raise ManifestError(
                    f"Invalid strategy optimization settings: {exc}"
                ) from exc
        except (TypeError, ValueError, StrategyNotExecutableError) as exc:
            return {
                "job_id": "",
                "message": str(exc),
                "total_combinations": 0,
                "code": "invalid_optimization_request",
            }
        job_id = uuid4().hex[:12]
        total_combinations = optimization.total_combinations
        self._job_store.create_job(
            job_id=job_id,
            job_type="optimization",
            strategy_name=strategy_id,
            revision_id=spec.revision_id,
            symbol=symbol,
            timeframe=timeframe,
            start_date=train_start,
            end_date=test_end,
            manifest=manifest,
            extra_payload={
                "original_config_path": str(config_path),
                "total_combinations": total_combinations,
            },
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
            "manifest_hash": manifest.manifest_hash,
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
