"""KD Stochastic Oscillator crossover trading strategy."""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING, Any

from btc_strategy.data.schemas import KDStrategyConfig
from btc_strategy.indicators.kd import KDIndicator
from btc_strategy.strategies.base import BaseStrategy
from btc_strategy.utils.logger import setup_logger

if TYPE_CHECKING:
    import pandas as pd

    from btc_strategy.backtest.base_engine import (
        BaseBacktestEngine,
    )
    from btc_strategy.data.schemas import KDFitConfig

logger = setup_logger(__name__)


class KDStrategy(BaseStrategy):
    """KD crossover strategy for BTC/USDT futures."""

    def __init__(
        self,
        config: KDStrategyConfig,
        fit_config: KDFitConfig | None = None,
        backtest_engine: BaseBacktestEngine | None = None,
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
        """Grid search over KD parameter combinations."""
        if self._fit_config is None:
            return

        if self._backtest_engine is None:
            msg = "backtest_engine is required for fit()"
            raise RuntimeError(msg)

        engine = self._backtest_engine
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

        for k_p, d_p, sm_k, ob, os_ in grid:
            trial_config = KDStrategyConfig(
                k_period=k_p,
                d_period=d_p,
                smooth_k=sm_k,
                overbought=ob,
                oversold=os_,
            )
            trial = KDStrategy(config=trial_config)
            signals = trial.generate_signals(df)
            metrics = engine.run(signals)
            value = metrics.get(target, float("-inf"))

            if isinstance(value, float) and value > best_metric:
                best_metric = value
                best_config = trial_config

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
        """Generate long/short signals based on KD crossovers."""
        result = self._indicator.calculate(df)

        k = result["stoch_k"]
        d = result["stoch_d"]
        k_prev = k.shift(1)
        d_prev = d.shift(1)

        golden_cross = (k > d) & (k_prev <= d_prev)
        death_cross = (k < d) & (k_prev >= d_prev)

        long_signal = golden_cross & (k < self._config.oversold)
        short_signal = death_cross & (k > self._config.overbought)

        result["signal"] = 0
        result.loc[long_signal, "signal"] = 1
        result.loc[short_signal, "signal"] = -1

        return result

    def get_parameters(self) -> dict[str, Any]:
        return self._config.model_dump()
