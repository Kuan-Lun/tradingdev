"""Application service for backtest execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from tradingdev.app.data_service import DataService
from tradingdev.domain.backtest.engines import create_backtest_engine
from tradingdev.domain.backtest.pipeline_result import PipelineResult
from tradingdev.domain.backtest.schemas import (
    BacktestConfig,
    BacktestRunConfig,
    ParallelConfig,
    WalkForwardConfig,
)
from tradingdev.domain.strategies.loader import StrategyLoader
from tradingdev.domain.validation.report import summarize_results
from tradingdev.domain.validation.walk_forward import WalkForwardValidator
from tradingdev.shared.utils.config import load_config

if TYPE_CHECKING:
    from pathlib import Path

    from tradingdev.domain.backtest.base_engine import BaseBacktestEngine


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
    ) -> None:
        self._data_service = data_service or DataService()
        self._strategy_loader = strategy_loader or StrategyLoader()

    def run_config(
        self, config_path: Path, *, walk_forward: bool = False
    ) -> BacktestRun:
        """Run a YAML config as simple backtest or walk-forward validation."""
        raw_config: dict[str, Any] = load_config(config_path)
        run_config = BacktestRunConfig.model_validate(raw_config)
        if run_config.is_walk_forward and not walk_forward:
            msg = "Config contains validation settings; use start_walk_forward."
            raise ValueError(msg)
        if walk_forward and not run_config.is_walk_forward:
            msg = "Config has no validation section for walk-forward."
            raise ValueError(msg)
        return self.run_raw_config(raw_config, walk_forward=walk_forward)

    def run_raw_config(
        self,
        raw_config: dict[str, Any],
        *,
        walk_forward: bool = False,
    ) -> BacktestRun:
        """Run an already parsed config."""
        bt_cfg = BacktestConfig(**raw_config["backtest"])
        parallel_cfg = ParallelConfig(**raw_config.get("parallel", {}))
        dataset = self._data_service.load(raw_config, bt_cfg)
        engine = self.create_engine(bt_cfg)
        strategy = self._strategy_loader.create_from_config(
            raw_config, engine, parallel_cfg
        )

        if walk_forward:
            wf_cfg = WalkForwardConfig(**raw_config["validation"])
            validator = WalkForwardValidator(config=wf_cfg, engine=engine)
            folds = validator.validate(strategy, dataset.frame)
            pipeline = PipelineResult(
                mode="walk_forward",
                fold_results=folds,
                config_snapshot=raw_config,
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
        )
        return BacktestRun(
            mode="simple",
            pipeline=pipeline,
            metrics=self.serialize_metrics(result.metrics),
            processed_path=dataset.processed_path,
            dataset_id=dataset.dataset_id,
        )

    def create_engine(self, config: BacktestConfig) -> BaseBacktestEngine:
        """Create a backtest engine from config."""
        return create_backtest_engine(config)

    def serialize_metrics(self, metrics: dict[str, Any]) -> dict[str, Any]:
        """Return the JSON-relevant metrics subset."""
        return {
            key: metrics[key] for key in self._RESULT_METRICS_KEYS if key in metrics
        }
