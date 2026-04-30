"""Abstract base class for backtest engines."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

    from tradingdev.backtest.result import BacktestResult


class BaseBacktestEngine(ABC):
    """Base interface for backtest execution engines.

    Subclasses implement different execution modes (signal-based,
    volume-based, etc.) while sharing common configuration.
    """

    def __init__(
        self,
        init_cash: float | None = None,
        fees: float = 0.0006,
        slippage: float = 0.0005,
        freq: str = "1h",
        position_size: float | None = None,
        stop_loss: float | None = None,
        take_profit: float | None = None,
        signal_as_position: bool = False,
        re_entry_after_sl: bool = True,
    ) -> None:
        self._init_cash = init_cash
        self._fees = fees
        self._slippage = slippage
        self._freq = freq
        self._position_size = position_size
        self._stop_loss = stop_loss
        self._take_profit = take_profit
        self._signal_as_position = signal_as_position
        self._re_entry_after_sl = re_entry_after_sl

    @abstractmethod
    def run(self, df: pd.DataFrame) -> BacktestResult:
        """Run backtest on a DataFrame with a ``signal`` column.

        Args:
            df: DataFrame with at least ``close`` and ``signal``.

        Returns:
            BacktestResult containing metrics, equity curve, and trades.
        """
        ...
