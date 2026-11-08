"""Static numeric ML contracts checked by strict Mypy, not runtime pytest."""

from __future__ import annotations

from typing import assert_type

import pandas as pd

from tradingdev.domain.ml.evaluation.metrics import forward_returns, rolling_ic
from tradingdev.domain.ml.evaluation.simulated_returns import simulate_strategy_returns
from tradingdev.domain.ml.evaluation.stratified import quantile_bins
from tradingdev.domain.ml.features.technical_features import compute_ta_indicators
from tradingdev.domain.ml.probabilities import class_probability


def numeric_results(prices: pd.Series[float], probabilities: pd.DataFrame) -> None:
    assert_type(rolling_ic(prices, prices), pd.Series[float])
    assert_type(forward_returns(prices, 1), pd.Series[float])
    assert_type(quantile_bins(prices, 3), pd.Series[int])
    assert_type(
        simulate_strategy_returns(prices, prices).long_short,
        pd.Series[float] | pd.Series[int],
    )
    assert_type(compute_ta_indicators(prices)["rsi_14"], pd.Series[float])
    assert_type(class_probability(probabilities, 1), pd.Series[float] | pd.Series[int])


def integer_inputs(prices: pd.Series[int]) -> None:
    assert_type(rolling_ic(prices, prices), pd.Series[float])
    assert_type(forward_returns(prices, 1), pd.Series[float])
    assert_type(quantile_bins(prices, 3), pd.Series[int])
    assert_type(compute_ta_indicators(prices)["rsi_14"], pd.Series[float])
    assert_type(
        simulate_strategy_returns(prices, prices).long_short,
        pd.Series[float] | pd.Series[int],
    )
