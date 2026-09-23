"""Application service for backtest execution."""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Protocol

from tradingdev.app.data_service import DataService
from tradingdev.app.job_config import bind_strategy_revision
from tradingdev.app.strategy_service import (
    StrategyNotExecutableError,
    StrategyService,
)
from tradingdev.domain.backtest.engines import create_backtest_engine
from tradingdev.domain.backtest.pipeline_result import PipelineResult
from tradingdev.domain.backtest.schemas import (
    BacktestConfig,
    BacktestRunConfig,
    ParallelConfig,
    WalkForwardConfig,
)
from tradingdev.domain.execution import ExecutionManifest, OptimizationSpec
from tradingdev.domain.strategies.loader import StrategyLoader
from tradingdev.domain.validation.report import summarize_results
from tradingdev.domain.validation.walk_forward import WalkForwardValidator
from tradingdev.shared.utils.config import load_config

if TYPE_CHECKING:
    from pathlib import Path

    from tradingdev.domain.backtest.base_engine import BaseBacktestEngine
    from tradingdev.domain.strategies.schemas import StrategySpec


class StrategyExecutionGate(Protocol):
    """Narrow interface for the strategy lifecycle execution gate."""

    def resolve_executable(
        self, strategy_id: str, revision_id: str | None = None
    ) -> StrategySpec:
        """Return the spec for an executable strategy or raise."""
        ...


@dataclass(frozen=True)
class BacktestRun:
    """Result and metadata from a backtest service run."""

    mode: str
    pipeline: PipelineResult
    metrics: dict[str, Any]
    processed_path: Path
    dataset_id: str


class BacktestService:
    """Run simple and walk-forward backtests through one service path."""

    _RESULT_METRICS_KEYS = [
        "total_return",
        "total_pnl",
        "annual_return",
        "sharpe_ratio",
        "max_drawdown",
        "win_rate",
        "profit_factor",
        "total_trades",
        "monthly_pnl_mean",
        "monthly_pnl_std",
        "monthly_pnl_min",
        "monthly_pnl_max",
        "monthly_pnl_median",
        "n_months",
        "monthly_trades_mean",
    ]

    def __init__(
        self,
        *,
        data_service: DataService | None = None,
        strategy_loader: StrategyLoader | None = None,
        strategy_gate: StrategyExecutionGate | None = None,
    ) -> None:
        self._data_service = data_service or DataService()
        self._strategy_loader = strategy_loader or StrategyLoader()
        self._strategy_gate = strategy_gate or StrategyService()

    def run_config(
        self, config_path: Path, *, walk_forward: bool = False
    ) -> BacktestRun:
        """Run a YAML config as simple backtest or walk-forward validation."""
        raw_config: dict[str, Any] = load_config(config_path)
        return self.run_raw_config(raw_config, walk_forward=walk_forward)

    def run_raw_config(
        self,
        raw_config: dict[str, Any],
        *,
        walk_forward: bool = False,
    ) -> BacktestRun:
        """Run an already parsed config.

        Raises:
            StrategyNotExecutableError: If the strategy has not reached
                runnable or promoted status, or the config's source_path does
                not match the registered strategy source.
        """
        manifest = self.prepare_execution(
            raw_config, kind="walk_forward" if walk_forward else "backtest"
        )
        return self.run_manifest(manifest)

    def prepare_execution(
        self,
        raw_config: dict[str, Any],
        *,
        kind: Literal["backtest", "walk_forward", "optimization"],
        optimization: OptimizationSpec | None = None,
    ) -> ExecutionManifest:
        """Bind the strategy and resolve application defaults before execution."""
        config = deepcopy(raw_config)
        run_config = BacktestRunConfig.model_validate(config)
        if run_config.is_walk_forward and kind == "backtest":
            msg = "Config contains validation settings; use start_walk_forward."
            raise ValueError(msg)
        if kind == "walk_forward" and not run_config.is_walk_forward:
            msg = "Config has no validation section for walk-forward."
            raise ValueError(msg)
        self.prepare_strategy(config)
        config["data"] = self._data_service.execution_config(
            config, run_config.backtest
        )
        return ExecutionManifest.create(
            kind=kind, config=config, optimization=optimization
        )

    def run_manifest(self, manifest: ExecutionManifest) -> BacktestRun:
        """Execute a verified specification without rebuilding its defaults."""
        manifest.verify()
        if manifest.kind not in {"backtest", "walk_forward"}:
            msg = "BacktestService cannot execute an optimization manifest"
            raise ValueError(msg)
        raw_config = manifest.config_copy()
        self.prepare_strategy(raw_config)
        bt_cfg = BacktestConfig(**raw_config["backtest"])
        parallel_cfg = ParallelConfig(**raw_config.get("parallel", {}))
        dataset = self._data_service.load(raw_config, bt_cfg)
        engine = self.create_engine(bt_cfg)
        strategy = self._strategy_loader.create_from_config(
            raw_config, engine, parallel_cfg
        )

        if manifest.kind == "walk_forward":
            wf_cfg = WalkForwardConfig(**raw_config["validation"])
            validator = WalkForwardValidator(config=wf_cfg, engine=engine)
            folds = validator.validate(strategy, dataset.frame)
            pipeline = PipelineResult(
                mode="walk_forward",
                fold_results=folds,
                config_snapshot=raw_config,
                execution_manifest=manifest,
            )
            return BacktestRun(
                mode="walk_forward",
                pipeline=pipeline,
                metrics=summarize_results(folds),
                processed_path=dataset.processed_path,
                dataset_id=dataset.dataset_id,
            )

        signals = strategy.generate_signals(dataset.frame)
        result = engine.run(signals)
        pipeline = PipelineResult(
            mode="simple",
            backtest_result=result,
            config_snapshot=raw_config,
            execution_manifest=manifest,
        )
        return BacktestRun(
            mode="simple",
            pipeline=pipeline,
            metrics=self.serialize_metrics(result.metrics),
            processed_path=dataset.processed_path,
            dataset_id=dataset.dataset_id,
        )

    def prepare_strategy(
        self,
        raw_config: dict[str, Any],
        *,
        allow_parameter_overrides: bool = False,
    ) -> None:
        """Validate a pinned strategy before backtest or optimization loading."""
        strategy_cfg = raw_config.get("strategy")
        if not isinstance(strategy_cfg, dict):
            msg = "strategy config must be a mapping"
            raise ValueError(msg)
        strategy_id = strategy_cfg.get("id")
        if not isinstance(strategy_id, str) or not strategy_id:
            msg = "strategy.id is required"
            raise ValueError(msg)
        revision_id = strategy_cfg.get("revision_id")
        if revision_id is not None and not isinstance(revision_id, str):
            msg = "strategy.revision_id must be a string"
            raise StrategyNotExecutableError(msg)
        spec = self._strategy_gate.resolve_executable(strategy_id, revision_id)
        bind_strategy_revision(
            raw_config, spec, allow_parameter_overrides=allow_parameter_overrides
        )

    def create_engine(self, config: BacktestConfig) -> BaseBacktestEngine:
        """Create a backtest engine from config."""
        return create_backtest_engine(config)

    def serialize_metrics(self, metrics: dict[str, Any]) -> dict[str, Any]:
        """Return JSON metrics with the same non-finite values as MCP responses."""
        serialized: dict[str, Any] = {}
        for key in self._RESULT_METRICS_KEYS:
            if key not in metrics:
                continue
            value = metrics[key]
            serialized[key] = (
                None if isinstance(value, float) and not math.isfinite(value) else value
            )
        return serialized
