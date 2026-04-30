"""Analytical computations for the dashboard."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd


def build_equity_series(
    equity_curve: npt.NDArray[np.float64],
    timestamps: npt.NDArray[Any] | None,
) -> pd.Series[float]:
    """Convert raw equity array to a pandas Series with datetime index."""
    if timestamps is not None:
        index: pd.DatetimeIndex | pd.RangeIndex = pd.DatetimeIndex(timestamps)
    else:
        index = pd.RangeIndex(len(equity_curve))
    return pd.Series(equity_curve, index=index, dtype=float)


def build_trades_df(
    trades: list[dict[str, Any]],
    timestamps: npt.NDArray[Any] | None = None,
) -> pd.DataFrame:
    """Convert trade list to DataFrame with an approximate timestamp.

    The timestamp is approximated from the trade index; if bar-level
    timestamps are not embedded in the trade records a simple integer
    index is used.
    """
    if not trades:
        return pd.DataFrame(
            columns=[
                "direction",
                "entry_price",
                "exit_price",
                "size_quote",
                "gross_pnl",
                "fee",
                "net_pnl",
            ]
        )
    df = pd.DataFrame(trades)
    if "timestamp" not in df.columns and timestamps is not None:
        # Spread trade indices evenly across the timestamp range
        n_ts = len(timestamps)
        n_trades = len(df)
        indices = np.linspace(0, n_ts - 1, n_trades, dtype=int)
        df["timestamp"] = timestamps[indices]
    return df


def cumulative_pnl(
    equity: pd.Series[float],
    init_cash: float | None,
) -> pd.Series[float]:
    """Compute cumulative PnL (absolute) from equity curve.

    When ``init_cash`` is ``None`` (volume mode), the equity curve
    already represents cumulative P&L from zero.
    """
    if init_cash is None:
        return equity
    return equity - init_cash


def cumulative_pnl_pct(
    equity: pd.Series[float],
    init_cash: float | None,
) -> pd.Series[float]:
    """Compute cumulative PnL as percentage.

    Returns a zero series when ``init_cash`` is ``None`` (volume mode)
    since percentage return is not meaningful without a capital base.
    """
    if init_cash is None or init_cash == 0:
        return pd.Series(0.0, index=equity.index)
    return (equity - init_cash) / init_cash * 100


def consecutive_loss_counts(
    trades_df: pd.DataFrame,
) -> pd.Series[int]:
    """Count frequencies of consecutive-loss streaks.

    Returns a Series where the index is the streak length and
    values are how many times that streak length occurred.
    """
    if trades_df.empty or "net_pnl" not in trades_df.columns:
        return pd.Series(dtype=int)

    is_loss = (trades_df["net_pnl"] <= 0).astype(int).values
    streaks: list[int] = []
    current = 0
    for v in is_loss:
        if v == 1:
            current += 1
        else:
            if current > 0:
                streaks.append(current)
            current = 0
    if current > 0:
        streaks.append(current)

    if not streaks:
        return pd.Series(dtype=int)

    s = pd.Series(streaks)
    counts = s.value_counts().sort_index()
    counts.index.name = "consecutive_losses"
    counts.name = "frequency"
    return counts


def rolling_mdd_absolute(
    equity: pd.Series[float],
    window_bars: int,
) -> pd.Series[float]:
    """Compute rolling maximum drawdown in absolute dollar amount.

    For each bar *i*, look back ``window_bars`` and compute::

        MDD_i = max(cummax[i-w:i] - equity[i-w:i])

    Returns a Series of the same length as *equity*, with NaN where
    the window is not yet full.
    """
    roll_max = equity.rolling(window=window_bars, min_periods=1).max()
    drawdown = roll_max - equity
    rolling_mdd = drawdown.rolling(window=window_bars, min_periods=window_bars).max()
    return rolling_mdd


def monthly_volume(
    trades_df: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate total traded volume by month.

    Returns DataFrame with columns ``month`` (str) and ``volume_quote``.
    """
    if trades_df.empty or "timestamp" not in trades_df.columns:
        return pd.DataFrame(columns=["month", "volume_quote"])

    df = trades_df.copy()
    df["month"] = pd.to_datetime(df["timestamp"]).dt.to_period("M")
    grouped = (
        df.groupby("month")["size_quote"]
        .sum()
        .multiply(2)  # round-trip volume
        .reset_index()
    )
    grouped.columns = pd.Index(["month", "volume_quote"])
    grouped["month"] = grouped["month"].astype(str)
    return pd.DataFrame(grouped)


def filter_by_month(
    equity: pd.Series[float],
    trades_df: pd.DataFrame,
    month: str | None,
) -> tuple[pd.Series[float], pd.DataFrame]:
    """Filter equity and trades to a specific month (YYYY-MM) or all."""
    if month is None or month == "全期間":
        return equity, trades_df

    if hasattr(equity.index, "to_period"):
        mask = equity.index.to_period("M").astype(str) == month
        eq_filtered = equity.loc[mask]
    else:
        eq_filtered = equity

    if not trades_df.empty and "timestamp" in trades_df.columns:
        ts = pd.to_datetime(trades_df["timestamp"])
        t_mask = ts.dt.to_period("M").astype(str) == month
        tr_filtered = trades_df.loc[t_mask]
    else:
        tr_filtered = trades_df

    return eq_filtered, tr_filtered


def available_months(
    timestamps: npt.NDArray[Any] | None,
) -> list[str]:
    """Return list of YYYY-MM strings present in timestamps."""
    if timestamps is None:
        return []
    idx = pd.DatetimeIndex(timestamps)
    months = sorted(idx.to_period("M").unique().astype(str).tolist())
    return months
