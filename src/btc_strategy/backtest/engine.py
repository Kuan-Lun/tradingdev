"""Vectorbt-based backtest execution engine."""

from typing import Any

import pandas as pd
import vectorbt as vbt

from btc_strategy.backtest.metrics import calculate_metrics
from btc_strategy.utils.logger import setup_logger

logger = setup_logger(__name__)


class BacktestEngine:
    """Execute backtests using vectorbt.

    Converts strategy signal columns (1 / -1 / 0) into boolean
    entry/exit arrays for vectorbt's ``Portfolio.from_signals``.

    The signal is shifted by 1 bar to avoid look-ahead bias: a signal
    generated at bar ``i``'s close is executed at bar ``i+1``'s open.
    """

    def __init__(
        self,
        init_cash: float = 10_000.0,
        fees: float = 0.0006,
        slippage: float = 0.0005,
        freq: str = "1h",
    ) -> None:
        self._init_cash = init_cash
        self._fees = fees
        self._slippage = slippage
        self._freq = freq

    def run(self, df: pd.DataFrame) -> dict[str, Any]:
        """Run backtest on a DataFrame with a ``signal`` column.

        Args:
            df: DataFrame with at least ``close`` and ``signal`` columns.
                Signal values: 1 (long), -1 (short), 0 (flat).

        Returns:
            Dictionary of performance metrics.
        """
        close = df["close"].astype(float)

        # Shift signals by 1 to avoid look-ahead bias
        signal = df["signal"].shift(1).fillna(0).astype(int)

        # Convert to boolean entry/exit arrays
        # Entry: transition into a position
        # Exit: transition out of a position
        entries = (signal == 1) & (signal.shift(1) != 1)
        exits = (signal != 1) & (signal.shift(1) == 1)
        short_entries = (signal == -1) & (signal.shift(1) != -1)
        short_exits = (signal != -1) & (signal.shift(1) == -1)

        logger.info(
            "Running backtest: init_cash=%.0f, fees=%.4f, slippage=%.4f",
            self._init_cash,
            self._fees,
            self._slippage,
        )

        pf = vbt.Portfolio.from_signals(
            close=close,
            entries=entries,
            exits=exits,
            short_entries=short_entries,
            short_exits=short_exits,
            init_cash=self._init_cash,
            fees=self._fees,
            slippage=self._slippage,
            freq=self._freq,
        )

        metrics = calculate_metrics(pf)
        logger.info("Backtest complete: %d trades", metrics["total_trades"])
        return metrics
