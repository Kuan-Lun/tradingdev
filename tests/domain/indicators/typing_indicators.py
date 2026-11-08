"""Static indicator contracts checked by strict Mypy, not runtime pytest."""

from __future__ import annotations

from typing import assert_type

import pandas as pd

from tradingdev.domain import indicators


def indicator_results(prices: pd.Series[float]) -> None:
    assert_type(indicators.sma(prices, 3), pd.Series[float])
    assert_type(indicators.macd(prices).histogram, pd.Series[float])
    # A regression to Any makes strict Mypy report this ignore as unused.
    _requires_text(indicators.sma(prices, 3).iloc[0])  # type: ignore[arg-type]


def integer_inputs(prices: pd.Series[int]) -> None:
    assert_type(indicators.sma(prices, 3), pd.Series[float])
    assert_type(indicators.atr(prices, prices, prices), pd.Series[float])


def _requires_text(value: str) -> None:
    pass
