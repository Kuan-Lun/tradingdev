"""KD Stochastic Oscillator crossover trading strategy."""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING, Any

from btc_strategy.backtest.engine import BacktestEngine
from btc_strategy.data.schemas import KDFitConfig, KDStrategyConfig
from btc_strategy.indicators.kd import KDIndicator
from btc_strategy.strategies.base import BaseStrategy
from btc_strategy.utils.logger import setup_logger

if TYPE_CHECKING:
    import pandas as pd

logger = setup_logger(__name__)


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

    def __init__(
        self,
        config: KDStrategyConfig,
        fit_config: KDFitConfig | None = None,
        backtest_engine: BacktestEngine | None = None,
    ) -> None:
        self._config = config
        self._fit_config = fit_config
        self._backtest_engine = backtest_engine
        self._indicator = KDIndicator(
            k_period=config.k_period,
            d_period=config.d_period,
            smooth_k=config.smooth_k,
        )

    def fit(self, df: pd.DataFrame) -> None:
        """Grid search over KD parameter combinations.

        For each combination, runs a full backtest on *df* and selects
        the parameters that maximise the configured target metric.

        Args:
            df: Training OHLCV DataFrame.
        """
        if self._fit_config is None:
            return

        engine = self._backtest_engine or BacktestEngine()
        target = self._fit_config.target_metric

        grid = list(
            itertools.product(
                self._fit_config.k_period_range,
                self._fit_config.d_period_range,
                self._fit_config.smooth_k_range,
                self._fit_config.overbought_range,
                self._fit_config.oversold_range,
            )
        )

        logger.info("KD grid search: %d combinations", len(grid))

        best_metric = -float("inf")
        best_config = self._config

        for k_period, d_period, smooth_k, overbought, oversold in grid:
            trial_config = KDStrategyConfig(
                k_period=k_period,
                d_period=d_period,
                smooth_k=smooth_k,
                overbought=overbought,
                oversold=oversold,
            )
            trial = KDStrategy(config=trial_config)
            signals = trial.generate_signals(df)
            metrics = engine.run(signals)
            value = metrics.get(target, float("-inf"))

            if isinstance(value, float) and value > best_metric:
                best_metric = value
                best_config = trial_config

        # Update self with best parameters
        self._config = best_config
        self._indicator = KDIndicator(
            k_period=best_config.k_period,
            d_period=best_config.d_period,
            smooth_k=best_config.smooth_k,
        )

        logger.info(
            "Best KD params: %s (%s=%.4f)",
            best_config.model_dump(),
            target,
            best_metric,
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
