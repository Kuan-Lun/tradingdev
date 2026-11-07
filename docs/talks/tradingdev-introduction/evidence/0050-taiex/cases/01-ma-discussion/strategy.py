"""Long-only SMA with persistent position on equality."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from tradingdev.domain import indicators
from tradingdev.domain.strategies.base import BaseStrategy

if TYPE_CHECKING:
    import pandas as pd


class SingleSmaHoldStrategy(BaseStrategy):
    """Enter above SMA, exit below it, retain state on equality."""

    def __init__(self, period: int = 20) -> None:
        self._period = period

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        closes = result["close"].to_numpy(dtype=float)
        averages = indicators.sma(result["close"], self._period).to_numpy(dtype=float)
        state = 0
        signals: list[int] = []
        for close, average in zip(closes, averages, strict=True):
            if not (np.isfinite(close) and np.isfinite(average)):
                state = 0
            elif close > average:
                state = 1
            elif close < average:
                state = 0
            signals.append(state)
        result["signal"] = signals
        return result

    def get_parameters(self) -> dict[str, Any]:
        return {"period": self._period}
