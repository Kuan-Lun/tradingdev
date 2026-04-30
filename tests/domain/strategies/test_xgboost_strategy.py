"""Tests for the XGBoost direction prediction strategy."""

import pandas as pd
import pytest

from tradingdev.domain.ml.schemas import XGBoostModelConfig
from tradingdev.domain.strategies.bundled.xgboost_strategy.strategy import (
    XGBoostStrategy,
)
from tradingdev.domain.strategies.schemas import XGBoostStrategyConfig


class TestXGBoostStrategy:
    def _make_config(self) -> XGBoostStrategyConfig:
        return XGBoostStrategyConfig(
            model=XGBoostModelConfig(
                n_estimators=20,
                max_depth=3,
            ),
            lookback_candidates=[6, 12],
            retrain_interval=50,
            validation_ratio=0.2,
        )

    def test_generate_signals_before_fit_raises(
        self, large_ohlcv_df: pd.DataFrame
    ) -> None:
        """generate_signals() before fit() should raise."""
        strategy = XGBoostStrategy(config=self._make_config())
        with pytest.raises(RuntimeError, match="not fitted"):
            strategy.generate_signals(large_ohlcv_df)

    def test_fit_selects_lookback(self, large_ohlcv_df: pd.DataFrame) -> None:
        """After fit(), best_lookback should be from candidates."""
        strategy = XGBoostStrategy(config=self._make_config())
        strategy.fit(large_ohlcv_df)
        params = strategy.get_parameters()
        assert params["best_lookback"] in [6, 12]

    def test_generate_signals_valid(self, large_ohlcv_df: pd.DataFrame) -> None:
        """After fit(), generate_signals produces valid signal column."""
        config = self._make_config()
        strategy = XGBoostStrategy(config=config)

        # Use first 800 for fit, last 200 for test
        fit_data = large_ohlcv_df.iloc[:800].copy()
        test_data = large_ohlcv_df.iloc[800:].copy()

        strategy.fit(fit_data)
        result = strategy.generate_signals(test_data)

        assert "signal" in result.columns
        unique = set(result["signal"].unique())
        assert unique.issubset({-1, 0, 1})

    def test_get_parameters(self) -> None:
        """get_parameters() should include config fields."""
        strategy = XGBoostStrategy(config=self._make_config())
        params = strategy.get_parameters()
        assert "lookback_candidates" in params
        assert "retrain_interval" in params
        assert params["best_lookback"] is None
