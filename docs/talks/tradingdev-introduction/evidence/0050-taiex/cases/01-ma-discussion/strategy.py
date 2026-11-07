"""Long-only SMA strategy retaining position on equality."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from tradingdev.domain import indicators
from tradingdev.domain.strategies.base import BaseStrategy

if TYPE_CHECKING:
    import pandas as pd


class SmaHoldStrategy(BaseStrategy):
    """Hold above SMA, exit below, and retain state on equality."""

    def __init__(self, period: int = 20) -> None:
        self._period = period

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        close = result["close"].to_numpy(dtype=np.float64)
        average = indicators.sma(result["close"], self._period).to_numpy(
            dtype=np.float64
        )
        signals = np.zeros(len(result), dtype=np.int64)
        state = 0
        for i in range(len(result)):
            if not (np.isfinite(close[i]) and np.isfinite(average[i])):
                continue
            if close[i] > average[i]:
                state = 1
            elif close[i] < average[i]:
                state = 0
            signals[i] = state
        result["signal"] = signals
        return result

    def get_parameters(self) -> dict[str, Any]:
        return {"period": self._period}
