"""Time scale shared by GLFT estimators, training and signal generation."""

from __future__ import annotations

from numbers import Number

import pandas as pd


def glft_bar_seconds(df: pd.DataFrame) -> float:
    """Resolve a fixed elapsed duration from an uncompressed input timeline.

    At least two timestamped bars are required. Missing/duplicate/unordered
    timestamps and unequal intervals are errors, not a new inferred frequency.
    A uniform input slice cannot establish whether its provider omitted every
    other candle; acquisition remains responsible for the requested frequency.
    """
    if "timestamp" not in df.columns or len(df) < 2:
        raise ValueError("GLFT requires at least two bars with a timestamp column")
    if pd.api.types.is_numeric_dtype(df["timestamp"]) or any(
        isinstance(value, Number) for value in df["timestamp"].array
    ):
        raise ValueError("GLFT timestamps must be datetimes, not numeric offsets")
    stamps = pd.DatetimeIndex(pd.to_datetime(df["timestamp"], utc=True))
    if stamps.hasnans or not stamps.is_monotonic_increasing or not stamps.is_unique:
        raise ValueError("GLFT timestamps must be nonmissing, unique and increasing")
    intervals = stamps[1:] - stamps[:-1]
    if not (intervals == intervals[0]).all():
        raise ValueError(
            "GLFT requires uniform bar intervals; missing bars or variable-duration "
            "candles cannot be treated as consecutive equal-duration bars"
        )
    return float(intervals[0] / pd.Timedelta(seconds=1))


def ensure_glft_fit_scale(bar_seconds: float, fitted_bar_seconds: float | None) -> None:
    """Do not reuse trained horizons or tuned windows at another frequency."""
    if fitted_bar_seconds is not None and bar_seconds != fitted_bar_seconds:
        raise ValueError(
            "GLFT input bar duration differs from the fitted duration; refit at "
            "the new frequency before generating signals"
        )
