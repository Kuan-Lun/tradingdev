"""Tests for the walk-forward validation framework."""

import pandas as pd

from btc_strategy.backtest.engine import BacktestEngine
from btc_strategy.data.schemas import KDStrategyConfig, WalkForwardConfig
from btc_strategy.strategies.kd_strategy import KDStrategy
from btc_strategy.validation.walk_forward import (
    WalkForwardValidator,
    format_walk_forward_report,
)


class TestWalkForwardValidator:
    def setup_method(self) -> None:
        self.engine = BacktestEngine(
            init_cash=10_000.0, fees=0.0, slippage=0.0
        )

    def test_single_explicit_split(
        self, walk_forward_ohlcv_df: pd.DataFrame
    ) -> None:
        """Explicit date split should return exactly one fold."""
        ts = walk_forward_ohlcv_df["timestamp"]
        mid = ts.iloc[len(ts) // 2]

        config = WalkForwardConfig(
            train_start=ts.iloc[0].to_pydatetime(),
            train_end=mid.to_pydatetime(),
            test_start=mid.to_pydatetime(),
            test_end=ts.iloc[-1].to_pydatetime(),
        )
        strategy = KDStrategy(config=KDStrategyConfig())
        validator = WalkForwardValidator(
            config=config, engine=self.engine
        )
        results = validator.validate(strategy, walk_forward_ohlcv_df)
        assert len(results) == 1

    def test_result_contains_metrics(
        self, walk_forward_ohlcv_df: pd.DataFrame
    ) -> None:
        """Each fold result should contain train and test metrics."""
        ts = walk_forward_ohlcv_df["timestamp"]
        mid = ts.iloc[len(ts) // 2]

        config = WalkForwardConfig(
            train_start=ts.iloc[0].to_pydatetime(),
            train_end=mid.to_pydatetime(),
            test_start=mid.to_pydatetime(),
            test_end=ts.iloc[-1].to_pydatetime(),
        )
        strategy = KDStrategy(config=KDStrategyConfig())
        validator = WalkForwardValidator(
            config=config, engine=self.engine
        )
        results = validator.validate(strategy, walk_forward_ohlcv_df)

        r = results[0]
        assert "total_return" in r.train_metrics
        assert "total_return" in r.test_metrics
        assert "sharpe_ratio" in r.test_metrics

    def test_strategy_params_recorded(
        self, walk_forward_ohlcv_df: pd.DataFrame
    ) -> None:
        """Result should capture strategy parameters."""
        ts = walk_forward_ohlcv_df["timestamp"]
        mid = ts.iloc[len(ts) // 2]

        config = WalkForwardConfig(
            train_start=ts.iloc[0].to_pydatetime(),
            train_end=mid.to_pydatetime(),
            test_start=mid.to_pydatetime(),
            test_end=ts.iloc[-1].to_pydatetime(),
        )
        strategy = KDStrategy(config=KDStrategyConfig())
        validator = WalkForwardValidator(
            config=config, engine=self.engine
        )
        results = validator.validate(strategy, walk_forward_ohlcv_df)
        assert "k_period" in results[0].strategy_params

    def test_auto_split_rolling(
        self, walk_forward_ohlcv_df: pd.DataFrame
    ) -> None:
        """Auto rolling split with n_splits=2 should return 2 folds."""
        config = WalkForwardConfig(
            n_splits=2, train_ratio=0.7, expanding=False
        )
        strategy = KDStrategy(config=KDStrategyConfig())
        validator = WalkForwardValidator(
            config=config, engine=self.engine
        )
        results = validator.validate(strategy, walk_forward_ohlcv_df)
        assert len(results) == 2

    def test_summary_aggregation(
        self, walk_forward_ohlcv_df: pd.DataFrame
    ) -> None:
        """summary() should return mean/std/min/max per metric."""
        config = WalkForwardConfig(
            n_splits=2, train_ratio=0.7, expanding=False
        )
        strategy = KDStrategy(config=KDStrategyConfig())
        validator = WalkForwardValidator(
            config=config, engine=self.engine
        )
        results = validator.validate(strategy, walk_forward_ohlcv_df)
        summary = validator.summary(results)

        assert summary["n_folds"] == 2
        assert "total_return" in summary
        assert "mean" in summary["total_return"]

    def test_format_report(
        self, walk_forward_ohlcv_df: pd.DataFrame
    ) -> None:
        """format_walk_forward_report should produce a string."""
        ts = walk_forward_ohlcv_df["timestamp"]
        mid = ts.iloc[len(ts) // 2]

        config = WalkForwardConfig(
            train_start=ts.iloc[0].to_pydatetime(),
            train_end=mid.to_pydatetime(),
            test_start=mid.to_pydatetime(),
            test_end=ts.iloc[-1].to_pydatetime(),
        )
        strategy = KDStrategy(config=KDStrategyConfig())
        validator = WalkForwardValidator(
            config=config, engine=self.engine
        )
        results = validator.validate(strategy, walk_forward_ohlcv_df)
        report = format_walk_forward_report(results)
        assert "Walk-Forward" in report
        assert "Fold 0" in report
