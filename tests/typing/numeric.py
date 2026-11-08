"""Static regressions checked by the repository's strict Mypy invocation."""

from __future__ import annotations

from typing import assert_type

import pandas as pd

from tradingdev.domain import indicators
from tradingdev.domain.ml.evaluation.metrics import forward_returns, rolling_ic
from tradingdev.domain.ml.evaluation.simulated_returns import simulate_strategy_returns
from tradingdev.domain.ml.evaluation.stratified import quantile_bins
from tradingdev.domain.ml.features.technical_features import compute_ta_indicators
from tradingdev.domain.ml.probabilities import class_probability


def numeric_results(prices: pd.Series[float], probabilities: pd.DataFrame) -> None:
    """Public numeric results must keep their element type across module calls."""
    assert_type(indicators.sma(prices, 3), pd.Series[float])
    assert_type(indicators.macd(prices).histogram, pd.Series[float])
    assert_type(rolling_ic(prices, prices), pd.Series[float])
    assert_type(forward_returns(prices, 1), pd.Series[float])
    assert_type(quantile_bins(prices, 3), pd.Series[int])
    assert_type(
        simulate_strategy_returns(prices, prices).long_short,
        pd.Series[float] | pd.Series[int],
    )
    assert_type(compute_ta_indicators(prices)["rsi_14"], pd.Series[float])
    assert_type(class_probability(probabilities, 1), pd.Series[float] | pd.Series[int])

    # If a result regresses to Any, strict Mypy reports this ignore as unused.
    _requires_text(indicators.sma(prices, 3).iloc[0])  # type: ignore[arg-type]


def _requires_text(value: str) -> None:
    pass


def integer_inputs(prices: pd.Series[int]) -> None:
    """Integer prices and labels remain valid numeric inputs."""
    assert_type(indicators.sma(prices, 3), pd.Series[float])
    assert_type(indicators.atr(prices, prices, prices), pd.Series[float])
    assert_type(rolling_ic(prices, prices), pd.Series[float])
    assert_type(forward_returns(prices, 1), pd.Series[float])
    assert_type(quantile_bins(prices, 3), pd.Series[int])
    assert_type(compute_ta_indicators(prices)["rsi_14"], pd.Series[float])
    assert_type(
        simulate_strategy_returns(prices, prices).long_short,
        pd.Series[float] | pd.Series[int],
    )
