"""Long-only SMA state strategy."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from tradingdev.domain import indicators
from tradingdev.domain.strategies.base import BaseStrategy

if TYPE_CHECKING:
    import pandas as pd


class SmaHoldStrategy(BaseStrategy):
    """Enter above SMA, exit below, preserve state on equality."""

    def __init__(self, period: int = 20) -> None:
        self._period = period

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        average = indicators.sma(result["close"], self._period)
        closes = result["close"].to_numpy(dtype=float)
        means = average.to_numpy(dtype=float)
        signals = np.zeros(len(result), dtype=int)
        state = 0
        for i in range(len(result)):
            if not (np.isfinite(closes[i]) and np.isfinite(means[i])):
                signals[i] = 0
                continue
            if closes[i] > means[i]:
                state = 1
            elif closes[i] < means[i]:
                state = 0
            signals[i] = state
        result["signal"] = signals
        return result

    def get_parameters(self) -> dict[str, Any]:
        return {"period": self._period}
