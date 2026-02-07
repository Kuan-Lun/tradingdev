"""KD Stochastic Oscillator crossover trading strategy."""

from typing import Any

import pandas as pd

from btc_strategy.data.schemas import KDStrategyConfig
from btc_strategy.indicators.kd import KDIndicator
from btc_strategy.strategies.base import BaseStrategy


class KDStrategy(BaseStrategy):
    """KD crossover strategy for BTC/USDT futures.

    Signal logic:
        - **Long (1)**: %K crosses above %D while both are below the oversold level.
        - **Short (-1)**: %K crosses below %D while both are above the overbought level.
        - **No signal (0)**: Otherwise.

    Crossover detection uses current and previous bar values.
    No look-ahead bias: signals at bar ``i`` only use data up to bar ``i``.
    The backtest engine applies a 1-bar shift before execution.
    """

    def __init__(self, config: KDStrategyConfig) -> None:
        self._config = config
        self._indicator = KDIndicator(
            k_period=config.k_period,
            d_period=config.d_period,
            smooth_k=config.smooth_k,
        )

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate long/short signals based on KD crossovers.

        Args:
            df: OHLCV DataFrame.

        Returns:
            DataFrame with ``stoch_k``, ``stoch_d``, and ``signal`` columns added.
        """
        result = self._indicator.calculate(df)

        k = result["stoch_k"]
        d = result["stoch_d"]
        k_prev = k.shift(1)
        d_prev = d.shift(1)

        # Golden cross: K crosses above D
        golden_cross = (k > d) & (k_prev <= d_prev)
        # Death cross: K crosses below D
        death_cross = (k < d) & (k_prev >= d_prev)

        # Long: golden cross in oversold zone
        long_signal = golden_cross & (k < self._config.oversold)
        # Short: death cross in overbought zone
        short_signal = death_cross & (k > self._config.overbought)

        result["signal"] = 0
        result.loc[long_signal, "signal"] = 1
        result.loc[short_signal, "signal"] = -1

        return result

    def get_parameters(self) -> dict[str, Any]:
        """Return strategy parameters."""
        return self._config.model_dump()
