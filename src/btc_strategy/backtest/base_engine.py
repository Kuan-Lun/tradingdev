"""Abstract base class for backtest engines."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd


class BaseBacktestEngine(ABC):
    """Base interface for backtest execution engines.

    Subclasses implement different execution modes (signal-based,
    volume-based, etc.) while sharing common configuration.
    """

    def __init__(
        self,
        init_cash: float = 10_000.0,
        fees: float = 0.0006,
        slippage: float = 0.0005,
        freq: str = "1h",
        position_size_usdt: float | None = None,
        stop_loss: float | None = None,
        take_profit: float | None = None,
    ) -> None:
        self._init_cash = init_cash
        self._fees = fees
        self._slippage = slippage
        self._freq = freq
        self._position_size_usdt = position_size_usdt
        self._stop_loss = stop_loss
        self._take_profit = take_profit

    @abstractmethod
    def run(self, df: pd.DataFrame) -> dict[str, Any]:
        """Run backtest on a DataFrame with a ``signal`` column.

        Args:
            df: DataFrame with at least ``close`` and ``signal``.

        Returns:
            Dictionary of performance metrics.
        """
        ...
