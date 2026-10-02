"""Break-even win rate and turnover cost computed only from trade records.

A strategy whose trades either win an average ``W`` or lose an average ``L``
breaks even when ``p * W = (1 - p) * L``, that is at the win rate
``p* = 1 / (1 + W / L) = L / (W + L)``. Comparing ``p*`` with the realized
win rate shows how much room the strategy has before its payoff profile
stops paying.

Conventions:

* A trade wins when its ``net_pnl`` is above zero and loses when it is below
  zero, as in :mod:`tradingdev.domain.statistics.performance`.
* Break-even trades contribute nothing to the expectancy, so the realized
  win rate compared with ``p*`` is measured over decisive trades only
  (``wins / (wins + losses)``). With this denominator the win-rate edge is
  positive exactly when the trades' total ``net_pnl`` is positive.
* Traded notional counts both legs of a round trip: the entry notional
  ``size_quote`` plus the exit notional ``size_quote * exit_price /
  entry_price``.
* Costs are the sum of the trades' ``fee`` fields. Figures are only as
  complete as those records: an engine that folds costs into ``net_pnl``
  without reporting them in ``fee`` understates the cost here.
* Undefined figures (no wins, no losses, no notional, no capital basis) are
  ``None`` rather than infinities or zeros.
"""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

DAYS_PER_YEAR = 365.25


@dataclass(frozen=True)
class BreakevenSummary:
    """Payoff profile, break-even win rate, and turnover cost of one run.

    Attributes:
        total_trades: Number of trade records.
        winning_trades: Trades with positive ``net_pnl``.
        losing_trades: Trades with negative ``net_pnl``.
        avg_win: Mean ``net_pnl`` of winning trades.
        avg_loss: Mean absolute ``net_pnl`` of losing trades.
        payoff_ratio: ``avg_win / avg_loss`` (``W / L``).
        breakeven_win_rate: ``1 / (1 + W / L)``, the decisive win rate at
            which the expectancy is zero.
        win_rate: Realized ``winning_trades / (winning_trades +
            losing_trades)``.
        win_rate_edge: ``win_rate - breakeven_win_rate``; positive means the
            strategy wins more often than its payoff profile requires.
        years: Length of the evaluation period in years of 365.25 days.
        traded_notional: Entry plus exit notional summed over all trades.
        annual_traded_notional: ``traded_notional / years``.
        annual_turnover: ``annual_traded_notional / capital``; ``None``
            without a capital basis.
        total_cost: Sum of the trades' ``fee`` fields.
        annual_cost: ``total_cost / years``.
        cost_per_turnover: ``total_cost / traded_notional``, the cost paid
            per unit of notional traded.
        annual_cost_drag: ``annual_cost / capital``, equal to
            ``annual_turnover * cost_per_turnover``.
    """

    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_win: float | None
    avg_loss: float | None
    payoff_ratio: float | None
    breakeven_win_rate: float | None
    win_rate: float | None
    win_rate_edge: float | None
    years: float
    traded_notional: float
    annual_traded_notional: float
    annual_turnover: float | None
    total_cost: float
    annual_cost: float
    cost_per_turnover: float | None
    annual_cost_drag: float | None


def summarize_breakeven(
    trades: Sequence[Mapping[str, Any]],
    *,
    start: Any,
    end: Any,
    capital: float | None = None,
) -> BreakevenSummary:
    """Compute the break-even win rate and turnover cost of a run's trades.

    Trade records carry no timestamps, so the evaluation period is passed
    explicitly, typically the first and last bar of the run.

    Args:
        trades: Trade records, each with numeric ``net_pnl``, ``size_quote``,
            ``entry_price``, ``exit_price``, and ``fee``.
        start: Start of the evaluation period; naive values are UTC.
        end: End of the evaluation period; naive values are UTC.
        capital: Capital basis for turnover, such as the initial cash.
            ``None`` (as in fixed-notional volume runs) leaves the
            capital-relative figures undefined.

    Raises:
        ValueError: If ``end`` is not after ``start``, ``capital`` is not
            positive, or a trade lacks a required finite field, has a
            negative size or fee, or has a non-positive entry or a negative
            exit price.
    """
    years = _years_between(start, end)
    if capital is not None and not (math.isfinite(capital) and capital > 0):
        raise ValueError("capital must be positive and finite")

    pnls: list[float] = []
    traded_notional = 0.0
    total_cost = 0.0
    for position, trade in enumerate(trades):
        pnls.append(_real_field(trade, "net_pnl", position))
        size = _real_field(trade, "size_quote", position)
        entry_price = _real_field(trade, "entry_price", position)
        exit_price = _real_field(trade, "exit_price", position)
        fee = _real_field(trade, "fee", position)
        if size < 0:
            raise ValueError(f"trade {position} has a negative size_quote")
        if entry_price <= 0 or exit_price < 0:
            raise ValueError(f"trade {position} has an invalid entry or exit price")
        if fee < 0:
            raise ValueError(f"trade {position} has a negative fee")
        traded_notional += size * (1.0 + exit_price / entry_price)
        total_cost += fee

    wins = [p for p in pnls if p > 0]
    losses = [-p for p in pnls if p < 0]
    avg_win = sum(wins) / len(wins) if wins else None
    avg_loss = sum(losses) / len(losses) if losses else None
    payoff_ratio = (
        avg_win / avg_loss if avg_win is not None and avg_loss is not None else None
    )
    breakeven_win_rate = (
        1.0 / (1.0 + payoff_ratio) if payoff_ratio is not None else None
    )
    decisive = len(wins) + len(losses)
    win_rate = len(wins) / decisive if decisive else None

    annual_traded_notional = traded_notional / years
    annual_cost = total_cost / years
    return BreakevenSummary(
        total_trades=len(pnls),
        winning_trades=len(wins),
        losing_trades=len(losses),
        avg_win=avg_win,
        avg_loss=avg_loss,
        payoff_ratio=payoff_ratio,
        breakeven_win_rate=breakeven_win_rate,
        win_rate=win_rate,
        win_rate_edge=(
            win_rate - breakeven_win_rate
            if win_rate is not None and breakeven_win_rate is not None
            else None
        ),
        years=years,
        traded_notional=traded_notional,
        annual_traded_notional=annual_traded_notional,
        annual_turnover=(
            annual_traded_notional / capital if capital is not None else None
        ),
        total_cost=total_cost,
        annual_cost=annual_cost,
        cost_per_turnover=(
            total_cost / traded_notional if traded_notional > 0 else None
        ),
        annual_cost_drag=annual_cost / capital if capital is not None else None,
    )


def _years_between(start: Any, end: Any) -> float:
    first = _utc_timestamp(start)
    last = _utc_timestamp(end)
    if last <= first:
        raise ValueError("end must be after start")
    return (last - first) / pd.Timedelta(days=DAYS_PER_YEAR)


def _utc_timestamp(value: Any) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def _real_field(trade: Mapping[str, Any], key: str, position: int) -> float:
    value = trade.get(key)
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"trade {position} has no numeric {key}")
    if not math.isfinite(value):
        raise ValueError(f"trade {position} has a non-finite {key}")
    return float(value)
