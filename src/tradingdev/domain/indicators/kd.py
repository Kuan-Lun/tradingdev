"""KD Stochastic Oscillator indicator."""

from typing import Any

import pandas as pd
import pandas_ta as ta

from tradingdev.domain.indicators.base import BaseIndicator


class KDIndicator(BaseIndicator):
    """Stochastic Oscillator (%K and %D) using pandas-ta.

    Appends ``stoch_k`` and ``stoch_d`` columns to the input DataFrame.
    """

    def __init__(
        self,
        k_period: int = 14,
        d_period: int = 3,
        smooth_k: int = 3,
    ) -> None:
        self._k_period = k_period
        self._d_period = d_period
        self._smooth_k = smooth_k

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate %K and %D values.

        Args:
            df: OHLCV DataFrame with ``high``, ``low``, ``close`` columns.

        Returns:
            DataFrame with ``stoch_k`` and ``stoch_d`` columns appended.
        """
        stoch = ta.stoch(
            high=df["high"],
            low=df["low"],
            close=df["close"],
            k=self._k_period,
            d=self._d_period,
            smooth_k=self._smooth_k,
        )

        result = df.copy()
        # pandas-ta returns columns like STOCHk_14_3_3, STOCHd_14_3_3
        result["stoch_k"] = stoch.iloc[:, 0]
        result["stoch_d"] = stoch.iloc[:, 1]
        return result

    def get_parameters(self) -> dict[str, Any]:
        """Return indicator parameters."""
        return {
            "k_period": self._k_period,
            "d_period": self._d_period,
            "smooth_k": self._smooth_k,
        }
