"""Backtest result container with raw data for analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

    from tradingdev.domain.backtest.execution_records import (
        AccountState,
        ExecutionRecord,
    )


@dataclass
class BacktestResult:
    """Holds both summary metrics and raw data from a backtest run.

    Attributes:
        metrics: Complete flat metric mapping; unavailable values are ``None``.
        equity_curve: Per-bar equity values.  For volume mode this is
            cumulative P&L starting from zero; for signal mode it is
            the portfolio value starting from ``init_cash``.
        trades: List of trade records, each a dict with keys like
            ``direction``, ``entry_price``, ``exit_price``, ``size_quote``,
            ``gross_pnl``, ``fee``, ``net_pnl``.
        timestamps: Per-bar timestamps (numpy datetime64 array), if available.
        init_cash: Starting capital.  ``None`` for volume mode.
        mode: Backtest mode (``"signal"`` or ``"volume"``).
        metric_metadata: Definitions version, provider versions, settings, and
            per-metric reasons for unavailable values.
        returns: Simple per-bar returns aligned to ``timestamps``; ``None``
            without a capital basis. Daily risk statistics aggregate these.
        execution_records: Native order attempts and account transitions; ``None``
            when not recorded, and an empty list when recorded without orders.
        account_history: Per-bar generic VectorBT cash, assets and equity; these
            are not exchange wallet balances or perpetual margin snapshots.
    """

    metrics: dict[str, Any]
    equity_curve: npt.NDArray[np.float64]
    trades: list[dict[str, Any]] = field(default_factory=list)
    timestamps: npt.NDArray[Any] | None = None
    init_cash: float | None = None
    mode: str = "signal"
    metric_metadata: dict[str, Any] = field(default_factory=dict)
    returns: npt.NDArray[np.float64] | None = None
    execution_records: list[ExecutionRecord] | None = None
    account_history: list[AccountState] | None = None
