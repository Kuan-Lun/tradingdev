"""GLFT volatility modes have distinct, stable numerical definitions."""

from __future__ import annotations

import math
from typing import Literal
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from tradingdev.domain.strategies.bundled.glft_ml_strategy.config import (
    GLFTMLStrategyConfig,
)
from tradingdev.domain.strategies.bundled.glft_ml_strategy.strategy import (
    GLFTMLStrategy,
)
from tradingdev.domain.strategies.bundled.glft_strategy.config import GLFTStrategyConfig
from tradingdev.domain.strategies.bundled.glft_strategy.strategy import GLFTStrategy
from tradingdev.domain.strategies.signals import generate_checked_signals


def _strategy(
    ml: bool, vol_type: Literal["realized", "parkinson", "implied"], window: int
) -> GLFTStrategy | GLFTMLStrategy:
    if ml:
        return GLFTMLStrategy(
            GLFTMLStrategyConfig(vol_type=vol_type, vol_window=window)
        )
    return GLFTStrategy(GLFTStrategyConfig(vol_type=vol_type, vol_window=window))


@pytest.mark.parametrize("ml", [False, True], ids=["glft", "glft_ml"])
def test_parkinson_uses_the_high_low_range_when_closes_are_constant(ml: bool) -> None:
    close = np.full(6, 100.0)
    high = np.full(6, 110.0)
    low = np.full(6, 90.0)
    parkinson = _strategy(ml, "parkinson", 3)
    realized = _strategy(ml, "realized", 3)

    sigma = parkinson._compute_volatility(close, high, low, bar_seconds=60)
    expected = math.log(110 / 90) / math.sqrt(4 * math.log(2))
    np.testing.assert_allclose(sigma, expected)
    actual_realized = realized._compute_volatility(close, high, low, bar_seconds=60)
    assert np.isnan(actual_realized[0])
    np.testing.assert_array_equal(actual_realized[1:], 1e-10)


@pytest.mark.parametrize("ml", [False, True], ids=["glft", "glft_ml"])
def test_parkinson_rolls_the_effective_window_and_preserves_unknown_values(
    ml: bool,
) -> None:
    strategy = _strategy(ml, "parkinson", 2 if ml else 5)
    # fit() can select a window different from the configured starting value.
    if isinstance(strategy, GLFTStrategy):
        strategy._best_vol_window = 2
    log_ranges = np.array([np.nan, 1.0, 2.0, np.nan, 3.0, np.nan, np.nan, 0.0])
    sigma = strategy._compute_volatility(
        np.ones(8), np.exp(log_ranges), np.ones(8), bar_seconds=60
    )
    # Independent means of the last two squared log ranges (skip missing).
    expected = np.sqrt(
        np.array([np.nan, 1.0, 2.5, 4.0, 9.0, 9.0, np.nan, 0.0]) / (4 * math.log(2))
    )
    expected[-1] = 1e-10
    np.testing.assert_allclose(sigma, expected, equal_nan=True)


@pytest.mark.parametrize("ml", [False, True], ids=["glft", "glft_ml"])
def test_realized_sample_deviation_and_warmup_are_preserved(ml: bool) -> None:
    strategy = _strategy(ml, "realized", 2)
    close = np.exp(np.array([0.0, 1.0, 3.0, 6.0]))
    result = strategy._compute_volatility(close, close, close, bar_seconds=60)
    np.testing.assert_allclose(result, [np.nan, *(3 * [math.sqrt(0.5)])])


@pytest.mark.parametrize("ml", [False, True], ids=["glft", "glft_ml"])
def test_implied_ignores_window_and_keeps_missing_volatility(ml: bool) -> None:
    strategy = _strategy(ml, "implied", 0)
    close = np.full(3, 100.0)
    dvol = np.array([50.0, np.nan, 0.0])
    result = strategy._compute_volatility(
        close, close, close, dvol=dvol, bar_seconds=60
    )
    np.testing.assert_allclose(result, [0.5 / math.sqrt(525_960), np.nan, 1e-10])
    with pytest.raises(ValueError, match="dvol array is required"):
        strategy._compute_volatility(close, close, close, bar_seconds=60)


@pytest.mark.parametrize("vol_type, enters", [("realized", True), ("parkinson", False)])
def test_glft_ml_search_and_prediction_use_the_selected_volatility(
    vol_type: Literal["realized", "parkinson"],
    enters: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=6, freq="min", tz="UTC"),
            "open": 100.0,
            "close": [100.0, 98.0, 100.0, 98.0, 100.0, 98.0],
            "high": 110.0,
            "low": 90.0,
            "volume": 10.0,
        }
    )
    engine = Mock()
    engine.run.return_value.metrics = {"total_volume": 1.0}
    strategy = GLFTMLStrategy(
        GLFTMLStrategyConfig(
            vol_type=vol_type,
            vol_window=2,
            gamma=10.0,
            gamma_candidates=[10.0],
            kappa=1000.0,
            kappa_candidates=[1000.0],
            ema_window=2,
            ema_window_candidates=[2],
            min_holding_bars=1,
            max_holding_bars=2,
            max_holding_bars_candidates=[2],
            min_entry_edge_candidates=[0.0012],
            profit_target_ratio=0.0,
            profit_target_ratio_candidates=[0.0],
            strategy_sl=0.0,
            confidence_threshold_candidates=[0.55],
        ),
        backtest_engine=engine,
    )
    # Replace AutoML and its feature preparation, not the GLFT calculation/search.
    monkeypatch.setattr(
        strategy._feature_eng, "transform", lambda df, **kwargs: df.copy()
    )
    train = Mock()
    monkeypatch.setattr(strategy._ml_model, "train", train)
    monkeypatch.setattr(
        strategy._ml_model,
        "predict_proba",
        lambda df: pd.DataFrame({0: 0.1, 1: 0.9}, index=df.index),
    )
    strategy.fit(frame)
    train.assert_called_once()
    assert strategy.get_parameters()["ml_trained"]
    result = generate_checked_signals(strategy, frame)
    engine.run.assert_called_once()
    searched = engine.run.call_args.args[0]
    pd.testing.assert_frame_equal(searched, result)
    # With nonzero gamma the wide high/low spread blocks otherwise eligible entries.
    assert bool(result["signal"].ne(0).any()) is enters
