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
        equity_curve: Per-bar equity values.
        trades: List of trade records, each a dict with keys like
            ``direction``, ``entry_price``, ``exit_price``, ``size_usdt``,
            ``gross_pnl``, ``fee``, ``net_pnl``.
        timestamps: Per-bar timestamps (numpy datetime64 array), if available.
        init_cash: Starting capital.
        mode: Backtest mode (``"signal"`` or ``"volume"``).
    """

    metrics: dict[str, Any]
    equity_curve: npt.NDArray[np.float64]
    trades: list[dict[str, Any]] = field(default_factory=list)
    timestamps: npt.NDArray[Any] | None = None
    init_cash: float = 10_000.0
    mode: str = "signal"
