"""Single-SMA long-only strategy, retaining position on equality."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from tradingdev.domain import indicators
from tradingdev.domain.strategies.base import BaseStrategy

if TYPE_CHECKING:
    import pandas as pd


class SmaComparisonStrategy(BaseStrategy):
    """Compare the completed close with its SMA, then trade next open."""

    def __init__(self, period: int = 20, buy_and_hold: bool = False) -> None:
        if (
            isinstance(period, bool)
            or not isinstance(period, int)
            or not 2 <= period <= 100000
        ):
            raise ValueError("period must be an integer between 2 and 100000")
        self._period = period
        self._buy_and_hold = buy_and_hold

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        if self._buy_and_hold:
            result["signal"] = 1
            return result
        average = indicators.sma(result["close"], self._period)
        closes = result["close"].to_numpy(dtype=float)
        averages = average.to_numpy(dtype=float)
        state = 0
        signals: list[int] = []
        for close, mean in zip(closes, averages, strict=True):
            if not (np.isfinite(close) and np.isfinite(mean)):
                state = 0
            elif close > mean:
                state = 1
            elif close < mean:
                state = 0
            signals.append(state)
        result["signal"] = signals
        return result

    def get_parameters(self) -> dict[str, Any]:
        return {"period": self._period, "buy_and_hold": self._buy_and_hold}
