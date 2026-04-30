"""Backtest result container with raw data for analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt


@dataclass
class BacktestResult:
    """Holds both summary metrics and raw data from a backtest run.

    Attributes:
        metrics: Dictionary of performance metrics (same keys as before).
        equity_curve: Per-bar equity values.  For volume mode this is
            cumulative P&L starting from zero; for signal mode it is
            the portfolio value starting from ``init_cash``.
        trades: List of trade records, each a dict with keys like
            ``direction``, ``entry_price``, ``exit_price``, ``size_quote``,
            ``gross_pnl``, ``fee``, ``net_pnl``.
        timestamps: Per-bar timestamps (numpy datetime64 array), if available.
        init_cash: Starting capital.  ``None`` for volume mode.
        mode: Backtest mode (``"signal"`` or ``"volume"``).
    """

    metrics: dict[str, Any]
    equity_curve: npt.NDArray[np.float64]
    trades: list[dict[str, Any]] = field(default_factory=list)
    timestamps: npt.NDArray[Any] | None = None
    init_cash: float | None = None
    mode: str = "signal"
