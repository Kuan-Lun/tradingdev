"""Technical indicator feature extraction functions."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas_ta as ta

if TYPE_CHECKING:
    import pandas as pd


def compute_sma_ratios(
    close: pd.Series,
    windows: list[int],
) -> dict[str, pd.Series]:
    """Compute close/SMA(N) ratios."""
    features: dict[str, pd.Series] = {}
    for w in windows:
        sma = close.rolling(w).mean()
        features[f"close_sma_ratio_{w}"] = close / sma
    return features


def compute_volume_features(
    volume: pd.Series,
    windows: list[int],
) -> dict[str, pd.Series]:
    """Compute volume change rate and volume/SMA ratios."""
    features: dict[str, pd.Series] = {"volume_change": volume.pct_change()}
    for w in windows:
        vol_sma = volume.rolling(w).mean()
        features[f"vol_sma_ratio_{w}"] = volume / vol_sma
    return features


def indicator_column(frame: pd.DataFrame, prefix: str) -> pd.Series:
    """Select the single pandas-ta output column whose name starts with *prefix*.

    pandas-ta encodes parameters in its column names (``BBL_20_2.0_2.0``,
    ``MACDh_12_26_9``) and the exact suffix changes between releases, so
    columns are selected by their stable prefix rather than by position.
    """
    matches = [
        str(column) for column in frame.columns if str(column).startswith(prefix)
    ]
    if len(matches) != 1:
        msg = (
            f"expected exactly one column starting with {prefix!r}, "
            f"found {matches!r} in {list(frame.columns)!r}"
        )
        raise KeyError(msg)
    return frame[matches[0]]


def macd_histogram(close: pd.Series) -> pd.Series | None:
    """Return the MACD histogram (MACD line minus signal line).

    Returns ``None`` when the series is too short for pandas-ta to compute it.
    """
    macd_df = ta.macd(close)
    if macd_df is None:
        return None
    return indicator_column(macd_df, "MACDh_")


def bollinger_bands(
    close: pd.Series,
    length: int,
) -> tuple[pd.Series, pd.Series, pd.Series] | None:
    """Return ``(lower, middle, upper)`` Bollinger Bands.

    Returns ``None`` when the series is too short for pandas-ta to compute them.
    """
    bbands = ta.bbands(close, length=length)
    if bbands is None:
        return None
    return (
        indicator_column(bbands, "BBL_"),
        indicator_column(bbands, "BBM_"),
        indicator_column(bbands, "BBU_"),
    )


def compute_ta_indicators(
    close: pd.Series,
) -> dict[str, pd.Series]:
    """Compute RSI, MACD histogram, and Bollinger %B."""
    features: dict[str, pd.Series] = {}

    rsi = ta.rsi(close, length=14)
    if rsi is not None:
        features["rsi_14"] = rsi

    macd_hist = macd_histogram(close)
    if macd_hist is not None:
        features["macd_hist"] = macd_hist

    bands = bollinger_bands(close, length=20)
    if bands is not None:
        lower, _, upper = bands
        band_width = upper - lower
        features["bb_pctb"] = (close - lower) / band_width.replace(0, float("nan"))

    return features
