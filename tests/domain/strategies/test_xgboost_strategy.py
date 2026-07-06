"""Tests for the XGBoost direction prediction strategy."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import pytest

from tradingdev.domain.ml.retrainer import RollingRetrainer
from tradingdev.domain.ml.schemas import XGBoostModelConfig
from tradingdev.domain.strategies.bundled.xgboost_strategy.config import (
    XGBoostStrategyConfig,
)
from tradingdev.domain.strategies.bundled.xgboost_strategy.strategy import (
    XGBoostStrategy,
)

if TYPE_CHECKING:
    from collections.abc import Callable


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

    def test_no_look_ahead_bias(
        self,
        large_ohlcv_df: pd.DataFrame,
        assert_no_look_ahead: Callable[..., None],
    ) -> None:
        """Rolling prediction may only use bars at or before each bar."""
        strategy = XGBoostStrategy(config=self._make_config())
        fit_data = large_ohlcv_df.iloc[:800].copy()
        test_data = large_ohlcv_df.iloc[800:].reset_index(drop=True)

        strategy.fit(fit_data)
        assert_no_look_ahead(strategy, test_data, check_points=[60, 120, 199])


class _StubFeatureEngineer:
    """Feature engineer stand-in that passes bars through unchanged."""

    def transform(self, df: pd.DataFrame, include_target: bool = False) -> pd.DataFrame:
        return df


class _StubDirectionModel:
    """Direction model emitting a scripted probability per prediction."""

    def __init__(self, probabilities: list[tuple[float, float]]) -> None:
        self._probabilities = probabilities
        self._calls = 0

    def predict_proba(self, _df: pd.DataFrame) -> pd.DataFrame:
        p_long, p_short = self._probabilities[self._calls]
        self._calls += 1
        return pd.DataFrame({1: [p_long], -1: [p_short]})


class TestRollingRetrainerSignalMapping:
    def test_probabilities_map_to_long_short_and_flat(self) -> None:
        """The tri-state mapping must emit exactly -1, 0, and 1."""
        model = _StubDirectionModel([(0.9, 0.1), (0.1, 0.9), (0.5, 0.5)])
        retrainer = RollingRetrainer(
            model_config=XGBoostModelConfig(),
            retrain_interval=1000,
            threshold=0.6,
            cooldown=0,
            lookback=1000,
        )
        test_df = pd.DataFrame({"close": [100.0, 101.0, 102.0]})
        train_df = pd.DataFrame({"close": [99.0, 99.5]})

        signals = retrainer.run(
            test_df,
            train_df,
            model,  # type: ignore[arg-type]
            _StubFeatureEngineer(),  # type: ignore[arg-type]
        )

        assert signals.tolist() == [1, -1, 0]
