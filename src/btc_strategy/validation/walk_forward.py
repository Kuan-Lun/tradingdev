"""Walk-forward validation for trading strategies."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from btc_strategy.utils.logger import setup_logger
from btc_strategy.validation.splitter import DataSplitter

if TYPE_CHECKING:
    from datetime import datetime

    import pandas as pd

    from btc_strategy.backtest.base_engine import (
        BaseBacktestEngine,
    )
    from btc_strategy.data.schemas import WalkForwardConfig
    from btc_strategy.strategies.base import BaseStrategy

logger = setup_logger(__name__)


@dataclass
class WalkForwardResult:
    """Result from a single walk-forward fold."""

    fold_index: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    train_metrics: dict[str, Any] = field(default_factory=dict)
    test_metrics: dict[str, Any] = field(default_factory=dict)
    strategy_params: dict[str, Any] = field(default_factory=dict)


class WalkForwardValidator:
    """Orchestrate walk-forward validation cycles."""

    def __init__(
        self,
        config: WalkForwardConfig,
        engine: BaseBacktestEngine,
    ) -> None:
        self._config = config
        self._engine = engine
        self._splitter = DataSplitter(config)

    def validate(
        self,
        strategy: BaseStrategy,
        df: pd.DataFrame,
    ) -> list[WalkForwardResult]:
        """Run walk-forward validation.

        Args:
            strategy: Strategy instance (mutated by fit()).
            df: Full OHLCV DataFrame covering all periods.

        Returns:
            List of WalkForwardResult, one per fold.
        """
        splits = self._splitter.split(df)
        results: list[WalkForwardResult] = []

        for i, (train_df, test_df) in enumerate(splits):
            result = self._run_fold(i, strategy, train_df, test_df)
            results.append(result)

        return results

    def _run_fold(
        self,
        fold_idx: int,
        strategy: BaseStrategy,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> WalkForwardResult:
        """Execute a single validation fold."""
        train_start = train_df["timestamp"].iloc[0]
        train_end = train_df["timestamp"].iloc[-1]
        test_start = test_df["timestamp"].iloc[0]
        test_end = test_df["timestamp"].iloc[-1]

        logger.info(
            "Fold %d: train [%s ~ %s] -> test [%s ~ %s]",
            fold_idx,
            train_start,
            train_end,
            test_start,
            test_end,
        )

        strategy.fit(train_df)

        train_signals = strategy.generate_signals(train_df)
        train_metrics = self._engine.run(train_signals)

        test_signals = strategy.generate_signals(test_df)
        test_metrics = self._engine.run(test_signals)

        logger.info(
            "Fold %d: train %s=%.4f, test %s=%.4f",
            fold_idx,
            self._config.target_metric,
            train_metrics.get(self._config.target_metric, float("nan")),
            self._config.target_metric,
            test_metrics.get(self._config.target_metric, float("nan")),
        )

        return WalkForwardResult(
            fold_index=fold_idx,
            train_start=train_start.to_pydatetime(),
            train_end=train_end.to_pydatetime(),
            test_start=test_start.to_pydatetime(),
            test_end=test_end.to_pydatetime(),
            train_metrics=train_metrics,
            test_metrics=test_metrics,
            strategy_params=strategy.get_parameters(),
        )
