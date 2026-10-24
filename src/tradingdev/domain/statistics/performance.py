"""Performance summary computed only from raw run outputs.

The summary reads the equity curve, bar timestamps, and trade records and
never consults an engine's ``metrics`` dictionary, so it can cross-check
whatever an engine reports. Daily and monthly figures follow the UTC period
boundaries defined in :mod:`tradingdev.domain.statistics.periods`.

Conventions:

* Standard deviations are sample standard deviations (``ddof=1``) and are
  ``None`` with fewer than two periods.
* Return-based figures are ``None`` unless every equity value is strictly
  positive; a cumulative P&L curve starting at zero has no return basis.
* A trade wins when its ``net_pnl`` is above zero and loses when it is below
  zero; break-even trades count toward ``total_trades`` only.
"""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from tradingdev.domain.statistics.periods import (
    equity_series,
    has_positive_equity,
    period_pnl,
    period_returns,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import numpy.typing as npt
    import pandas as pd

DEFAULT_PERIODS_PER_YEAR = 365.0


@dataclass(frozen=True)
class PerformanceSummary:
    """Engine-independent performance overview of one run.

    Attributes:
        start: First timestamp, in UTC.
        end: Last timestamp, in UTC.
        n_bars: Number of equity observations.
        n_days: Number of UTC days containing at least one observation.
        n_months: Number of UTC months containing at least one observation.
        total_pnl: Last equity minus first equity.
        total_return: ``total_pnl`` relative to the first equity.
        max_drawdown: Largest peak-to-trough equity decline, as a positive
            absolute amount.
        max_drawdown_pct: Largest peak-to-trough decline relative to the peak.
        daily_pnl_mean: Mean daily P&L.
        daily_pnl_std: Sample standard deviation of daily P&L.
        monthly_pnl_mean: Mean monthly P&L.
        monthly_pnl_std: Sample standard deviation of monthly P&L.
        sharpe_ratio: Mean over sample standard deviation of daily returns,
            scaled by ``sqrt(periods_per_year)``; zero risk-free rate.
        total_trades: Number of trade records.
        winning_trades: Trades with positive ``net_pnl``.
        losing_trades: Trades with negative ``net_pnl``.
        win_rate: ``winning_trades / total_trades``.
        profit_factor: Gross winning P&L over gross losing P&L; ``None``
            when there is no losing P&L.
        trade_net_pnl: Sum of ``net_pnl`` over all trades.
    """

    start: pd.Timestamp
    end: pd.Timestamp
    n_bars: int
    n_days: int
    n_months: int
    total_pnl: float
    total_return: float | None
    max_drawdown: float
    max_drawdown_pct: float | None
    daily_pnl_mean: float
    daily_pnl_std: float | None
    monthly_pnl_mean: float
    monthly_pnl_std: float | None
    sharpe_ratio: float | None
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float | None
    profit_factor: float | None
    trade_net_pnl: float


def summarize_performance(
    equity_curve: npt.ArrayLike,
    timestamps: npt.ArrayLike,
    trades: Sequence[Mapping[str, Any]],
    *,
    periods_per_year: float = DEFAULT_PERIODS_PER_YEAR,
) -> PerformanceSummary:
    """Summarize a run from its equity curve, timestamps, and trades.

    Args:
        equity_curve: Per-bar equity values.
        timestamps: Per-bar timestamps; naive values are treated as UTC.
        trades: Trade records, each with a numeric ``net_pnl``.
        periods_per_year: Daily periods per year used to annualize the
            Sharpe ratio; 365 suits markets that trade every day.

    Raises:
        ValueError: If the inputs are inconsistent (see
            :func:`~tradingdev.domain.statistics.periods.equity_series`),
            ``periods_per_year`` is not positive, or a trade lacks a finite
            ``net_pnl``.
    """
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive")
    equity = equity_series(equity_curve, timestamps)
    daily = period_pnl(equity_curve, timestamps, "daily")
    monthly = period_pnl(equity_curve, timestamps, "monthly")
    positive = has_positive_equity(equity_curve)

    first = float(equity.iloc[0])
    total_pnl = float(equity.iloc[-1]) - first
    peaks = equity.cummax()
    trade_pnls = _trade_net_pnls(trades)
    gross_profit = sum(p for p in trade_pnls if p > 0)
    gross_loss = -sum(p for p in trade_pnls if p < 0)
    winning = sum(1 for p in trade_pnls if p > 0)

    return PerformanceSummary(
        start=equity.index[0],
        end=equity.index[-1],
        n_bars=len(equity),
        n_days=len(daily),
        n_months=len(monthly),
        total_pnl=total_pnl,
        total_return=total_pnl / first if positive else None,
        max_drawdown=float((peaks - equity).max()),
        max_drawdown_pct=float((1.0 - equity / peaks).max()) if positive else None,
        daily_pnl_mean=float(daily.mean()),
        daily_pnl_std=_sample_std(daily),
        monthly_pnl_mean=float(monthly.mean()),
        monthly_pnl_std=_sample_std(monthly),
        sharpe_ratio=(
            _sharpe(period_returns(equity_curve, timestamps, "daily"), periods_per_year)
            if positive
            else None
        ),
        total_trades=len(trade_pnls),
        winning_trades=winning,
        losing_trades=sum(1 for p in trade_pnls if p < 0),
        win_rate=winning / len(trade_pnls) if trade_pnls else None,
        profit_factor=gross_profit / gross_loss if gross_loss > 0 else None,
        trade_net_pnl=float(sum(trade_pnls)),
    )


def _trade_net_pnls(trades: Sequence[Mapping[str, Any]]) -> list[float]:
    pnls: list[float] = []
    for position, trade in enumerate(trades):
        value = trade.get("net_pnl")
        if isinstance(value, bool) or not isinstance(value, numbers.Real):
            raise ValueError(f"trade {position} has no numeric net_pnl")
        if not math.isfinite(value):
            raise ValueError(f"trade {position} has a non-finite net_pnl")
        pnls.append(float(value))
    return pnls


def _sample_std(values: pd.Series) -> float | None:
    if len(values) < 2:
        return None
    return float(values.std(ddof=1))


def _sharpe(daily_returns: pd.Series, periods_per_year: float) -> float | None:
    std = _sample_std(daily_returns)
    if std is None or std == 0.0:
        return None
    return float(daily_returns.mean()) / std * float(np.sqrt(periods_per_year))
