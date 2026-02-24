"""Abstract base class for all trading strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd


class BaseStrategy(ABC):
    """Base interface for trading strategies.

    Strategies receive a DataFrame (optionally with indicator columns)
    and produce a ``signal`` column with values:
    - ``1``:  Long entry
    - ``-1``: Short entry
    - ``0``:  No signal / flat
    """

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals.

        Args:
            df: OHLCV DataFrame (may already contain indicator columns).

        Returns:
            DataFrame with a ``signal`` column added.
        """
        ...

    def fit(self, df: pd.DataFrame) -> None:  # noqa: B027
        """Fit/optimize the strategy on training data.

        Default implementation is a no-op. Override in subclasses
        that support parameter optimization or model training.

        Args:
            df: Training OHLCV DataFrame.
        """

    @abstractmethod
    def get_parameters(self) -> dict[str, Any]:
        """Return strategy parameter dictionary.

        Returns:
            Dictionary of parameter names to values.
        """
        ...
