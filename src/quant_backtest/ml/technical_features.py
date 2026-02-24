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


def compute_ta_indicators(
    close: pd.Series,
) -> dict[str, pd.Series]:
    """Compute RSI, MACD histogram, and Bollinger %B."""
    features: dict[str, pd.Series] = {}

    rsi = ta.rsi(close, length=14)
    if rsi is not None:
        features["rsi_14"] = rsi

    macd_df = ta.macd(close)
    if macd_df is not None:
        features["macd_hist"] = macd_df.iloc[:, 2]

    bbands = ta.bbands(close, length=20)
    if bbands is not None:
        upper = bbands.iloc[:, 0]
        lower = bbands.iloc[:, 2]
        band_width = upper - lower
        features["bb_pctb"] = (close - lower) / band_width.replace(0, float("nan"))

    return features
