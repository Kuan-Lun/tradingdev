from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from tradingdev.domain.indicators import sma
from tradingdev.domain.strategies.base import BaseStrategy

if TYPE_CHECKING:
    import pandas as pd


class MovingAverageStrategy(BaseStrategy):
    def __init__(self, fast_period: int = 5, slow_period: int = 20) -> None:
        if not 2 <= fast_period < slow_period <= 100000:
            raise ValueError("Require 2 <= fast_period < slow_period <= 100000")
        self.fast_period = fast_period
        self.slow_period = slow_period

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        fast = sma(df["close"], length=self.fast_period)
        slow = sma(df["close"], length=self.slow_period)
        ready = np.isfinite(fast) & np.isfinite(slow) & np.isfinite(df["close"])
        result["signal"] = (ready & (fast > slow)).astype(int)
        return result

    def get_parameters(self) -> dict[str, Any]:
        return {"fast_period": self.fast_period, "slow_period": self.slow_period}
