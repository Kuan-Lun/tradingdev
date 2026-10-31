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

import numbers
from typing import Literal

import numpy as np
import numpy.typing as npt
import pandas as pd

from tradingdev.domain.statistics._validation import real_array

Frequency = Literal["daily", "monthly"]

_RESAMPLE_RULES: dict[Frequency, str] = {"daily": "D", "monthly": "MS"}
_NUMERIC_TIMESTAMPS = (
    "timestamps must be datetimes or date strings, not numbers ({}); a number "
    "has no unit, so convert it first, e.g. pd.to_datetime(values, unit='s')"
)


def to_utc_index(timestamps: npt.ArrayLike) -> pd.DatetimeIndex:
    """Return ``timestamps`` as a UTC ``DatetimeIndex`` (naive means UTC).

    Aware timestamps may carry different UTC offsets, as local times on
    either side of a daylight saving change do; each is converted to UTC on
    its own. Naive and aware timestamps in one sequence are rejected rather
    than guessing that the naive ones are UTC.

    The index is always in nanoseconds. pandas 3 keeps the resolution of its
    input (``date_range`` yields microseconds, a ``datetime64[ns]`` array
    nanoseconds), so without this the same timestamps would give period
    series with different index dtypes depending on how they were passed.

    Numbers are rejected, not read as times. A number carries no unit:
    pandas reads ``1_704_067_200`` as nanoseconds after the epoch, so Unix
    seconds silently became a fraction of a second on 1970-01-01, and 30
    days of hourly bars one daily period. There is no ``unit`` parameter on
    purpose: every function that takes timestamps would have to carry it,
    seconds and nanoseconds cannot be told apart reliably from the values,
    and the caller can state the unit once, for example with
    ``pd.to_datetime(values, unit="s")``. Run outputs already hold datetimes.

    Raises:
        ValueError: If ``timestamps`` is not one-dimensional, holds numbers
            or booleans, mixes naive and aware timestamps, or has a value
            that cannot be parsed as a timestamp.
    """
    if isinstance(timestamps, pd.DatetimeIndex):
        index = timestamps
    else:
        values = np.asarray(timestamps)
        if values.ndim != 1:
            raise ValueError("timestamps must be one-dimensional")
        if values.dtype.kind in "biufc":
            raise ValueError(_NUMERIC_TIMESTAMPS.format(values.dtype))
        index = (
            _parse_each(values)
            if values.dtype.kind in "OU"
            else pd.DatetimeIndex(values)
        )
    utc = index.tz_localize("UTC") if index.tz is None else index.tz_convert("UTC")
    return utc.as_unit("ns")


def _parse_each(values: npt.NDArray[np.object_ | np.str_]) -> pd.DatetimeIndex:
    # Each value is parsed on its own, as pd.DatetimeIndex did: one inferred
    # format for all (pd.to_datetime) rejects "2024-01-01" next to
    # "2024-01-01 01:00", and utc=True would silently read naive values as
    # UTC. Converting aware values one by one accepts differing offsets.
    elements = values.tolist()
    for element in elements:
        if isinstance(element, numbers.Number):
            raise ValueError(_NUMERIC_TIMESTAMPS.format(type(element).__name__))
    parsed = [pd.Timestamp(value) for value in elements]
    if len({ts.tzinfo is not None for ts in parsed if not pd.isna(ts)}) > 1:
        raise ValueError("timestamps mix timezone-naive and timezone-aware values")
    return pd.DatetimeIndex(
        [ts.tz_convert("UTC") if ts.tzinfo is not None else ts for ts in parsed]
    )


def equity_series(
    equity_curve: npt.ArrayLike,
    timestamps: npt.ArrayLike,
) -> pd.Series:
    """Validate raw run outputs and pair them into a UTC-indexed series.

    Raises:
        ValueError: If the inputs are empty, differ in length, equity is not
            numeric (booleans and strings are rejected, not converted) or not
            finite, or timestamps are invalid as for :func:`to_utc_index` or
            not strictly increasing.
    """
    equity = real_array(equity_curve, "equity_curve")
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

    Raises:
        ValueError: If ``frequency`` is not ``"daily"`` or ``"monthly"``, or
            the inputs are invalid as for :func:`equity_series`.
    """
    # Literal only constrains literal arguments; a str read from settings
    # would otherwise surface as a KeyError from the lookup below.
    if frequency not in _RESAMPLE_RULES:
        raise ValueError(
            f"frequency must be one of {sorted(_RESAMPLE_RULES)}, got {frequency!r}"
        )
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
    """Whether every equity value is strictly positive, so returns exist.

    Raises:
        ValueError: If ``equity_curve`` holds booleans, strings or other
            non-numbers, as for :func:`equity_series`.
    """
    return bool((real_array(equity_curve, "equity_curve") > 0.0).all())


def _opening_equity(closing: pd.Series, equity_curve: npt.ArrayLike) -> pd.Series:
    first = float(np.asarray(equity_curve, dtype=np.float64)[0])
    return closing.shift(1).fillna(first)
