"""Resample an equity curve into daily and monthly P&L and return series.

Period boundaries are UTC calendar boundaries:

* A day is the half-open interval ``[00:00:00, 24:00:00)`` UTC.
* A month is the half-open interval from ``00:00:00`` UTC on its first day
  to ``00:00:00`` UTC on the first day of the next month.

Every equity observation belongs to the period that contains its timestamp
label, exactly as given. Bar data labelled by open time (as on Binance) thus
assigns the ``23:00`` hourly bar to the day it opens in. Naive timestamps are
interpreted as UTC; timezone-aware timestamps are converted to UTC before
bucketing, so a local timezone never shifts the boundaries.

A period's closing equity is the last observation inside it. Its P&L is that
closing equity minus the previous period's closing equity; the first period
is measured against the first observation of the curve. Periods without any
observation (weekends on equity markets, data gaps) are omitted rather than
filled with zero. Period P&L therefore always sums to
``equity_curve[-1] - equity_curve[0]``.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import numpy.typing as npt
import pandas as pd

Frequency = Literal["daily", "monthly"]

_RESAMPLE_RULES: dict[Frequency, str] = {"daily": "D", "monthly": "MS"}


def to_utc_index(timestamps: npt.ArrayLike) -> pd.DatetimeIndex:
    """Return ``timestamps`` as a UTC ``DatetimeIndex`` (naive means UTC)."""
    index = (
        timestamps
        if isinstance(timestamps, pd.DatetimeIndex)
        else pd.DatetimeIndex(np.asarray(timestamps))
    )
    if index.tz is None:
        return index.tz_localize("UTC")
    return index.tz_convert("UTC")


def equity_series(
    equity_curve: npt.ArrayLike,
    timestamps: npt.ArrayLike,
) -> pd.Series:
    """Validate raw run outputs and pair them into a UTC-indexed series.

    Raises:
        ValueError: If the inputs are empty, differ in length, contain
            non-finite equity, or timestamps are not strictly increasing.
    """
    equity = np.asarray(equity_curve, dtype=np.float64)
    if equity.ndim != 1 or equity.size == 0:
        raise ValueError("equity_curve must be a non-empty one-dimensional array")
    if not np.isfinite(equity).all():
        raise ValueError("equity_curve must contain only finite values")
    index = to_utc_index(timestamps)
    if len(index) != len(equity):
        raise ValueError(
            f"timestamps length {len(index)} does not match "
            f"equity_curve length {len(equity)}"
        )
    if not (index.is_monotonic_increasing and index.is_unique):
        raise ValueError("timestamps must be strictly increasing")
    return pd.Series(equity, index=index, name="equity")


def period_closing_equity(
    equity_curve: npt.ArrayLike,
    timestamps: npt.ArrayLike,
    frequency: Frequency,
) -> pd.Series:
    """Closing equity of each UTC period that contains an observation.

    The index holds each period's UTC start (midnight of the day or of the
    month's first day).
    """
    series = equity_series(equity_curve, timestamps)
    closing = series.resample(_RESAMPLE_RULES[frequency]).last().dropna()
    closing.name = "equity"
    return closing


def period_pnl(
    equity_curve: npt.ArrayLike,
    timestamps: npt.ArrayLike,
    frequency: Frequency,
) -> pd.Series:
    """Absolute equity change of each UTC period."""
    closing = period_closing_equity(equity_curve, timestamps, frequency)
    pnl = closing - _opening_equity(closing, equity_curve)
    pnl.name = "pnl"
    return pnl


def period_returns(
    equity_curve: npt.ArrayLike,
    timestamps: npt.ArrayLike,
    frequency: Frequency,
) -> pd.Series:
    """Simple return of each UTC period relative to its opening equity.

    Compounding the returns reproduces ``equity_curve[-1] / equity_curve[0]``.

    Raises:
        ValueError: If any equity value is not strictly positive, as with a
            cumulative P&L curve starting at zero, where returns are undefined.
    """
    closing = period_closing_equity(equity_curve, timestamps, frequency)
    if not has_positive_equity(equity_curve):
        raise ValueError("period returns require strictly positive equity")
    returns = closing / _opening_equity(closing, equity_curve) - 1.0
    returns.name = "return"
    return returns


def has_positive_equity(equity_curve: npt.ArrayLike) -> bool:
    """Whether every equity value is strictly positive, so returns exist."""
    return bool((np.asarray(equity_curve, dtype=np.float64) > 0.0).all())


def _opening_equity(closing: pd.Series, equity_curve: npt.ArrayLike) -> pd.Series:
    first = float(np.asarray(equity_curve, dtype=np.float64)[0])
    return closing.shift(1).fillna(first)
