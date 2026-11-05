"""GLFT volatility modes have distinct, stable numerical definitions."""

from __future__ import annotations

import math

import numpy as np
import pytest

from tradingdev.domain.strategies.bundled.glft_strategy.config import GLFTStrategyConfig
from tradingdev.domain.strategies.bundled.glft_strategy.strategy import GLFTStrategy


def test_parkinson_uses_the_high_low_range_when_closes_are_constant() -> None:
    close = np.full(6, 100.0)
    high = np.full(6, 110.0)
    low = np.full(6, 90.0)
    parkinson = GLFTStrategy(GLFTStrategyConfig(vol_type="parkinson", vol_window=3))
    realized = GLFTStrategy(GLFTStrategyConfig(vol_type="realized", vol_window=3))

    sigma = parkinson._compute_volatility(close, high, low)
    expected = math.log(110 / 90) / math.sqrt(4 * math.log(2))
    np.testing.assert_allclose(sigma, expected)
    actual_realized = realized._compute_volatility(close, high, low)
    assert np.isnan(actual_realized[0])
    np.testing.assert_array_equal(actual_realized[1:], 1e-10)


def test_parkinson_rolls_the_effective_window_and_preserves_unknown_values() -> None:
    strategy = GLFTStrategy(GLFTStrategyConfig(vol_type="parkinson", vol_window=5))
    # fit() can select a window different from the configured starting value.
    strategy._best_vol_window = 2
    log_ranges = np.array([np.nan, 1.0, 2.0, np.nan, 3.0, np.nan, np.nan, 0.0])
    sigma = strategy._compute_volatility(np.ones(8), np.exp(log_ranges), np.ones(8))
    # Independent means of the last two squared log ranges (skip missing).
    expected = np.sqrt(
        np.array([np.nan, 1.0, 2.5, 4.0, 9.0, 9.0, np.nan, 0.0]) / (4 * math.log(2))
    )
    expected[-1] = 1e-10
    np.testing.assert_allclose(sigma, expected, equal_nan=True)


def test_realized_sample_deviation_and_warmup_are_preserved() -> None:
    strategy = GLFTStrategy(GLFTStrategyConfig(vol_type="realized", vol_window=2))
    close = np.exp(np.array([0.0, 1.0, 3.0, 6.0]))
    result = strategy._compute_volatility(close, close, close)
    np.testing.assert_allclose(result, [np.nan, *(3 * [math.sqrt(0.5)])])


def test_implied_ignores_window_and_keeps_missing_volatility() -> None:
    strategy = GLFTStrategy(GLFTStrategyConfig(vol_type="implied", vol_window=0))
    close = np.full(3, 100.0)
    dvol = np.array([50.0, np.nan, 0.0])
    result = strategy._compute_volatility(close, close, close, dvol=dvol)
    np.testing.assert_allclose(result, [0.5 / math.sqrt(525_960), np.nan, 1e-10])
    with pytest.raises(ValueError, match="dvol array is required"):
        strategy._compute_volatility(close, close, close)
