"""Named technical indicators backed by pandas-ta.

This module is the only place in the codebase that calls pandas-ta. Every
function selects pandas-ta output by column name, never by position, and pins
``talib=False`` so results do not depend on whether TA-Lib happens to be
installed. Inputs shorter than an indicator's window return NaN-filled series
of the same length, matching pandas rolling semantics, instead of pandas-ta's
``None``.

Where pandas-ta offers a choice, conventions follow TA-Lib: EMA is seeded with
the SMA of the first ``length`` bars, RSI, ATR and ADX use Wilder smoothing,
and Bollinger Bands use the population standard deviation (``ddof=0``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pandas_ta as ta


@dataclass(frozen=True)
class MACDResult:
    """MACD line, signal line, and histogram (``macd - signal``)."""

    macd: pd.Series
    signal: pd.Series
    histogram: pd.Series


@dataclass(frozen=True)
class BollingerBands:
    """Lower, middle (SMA), and upper Bollinger Bands."""

    lower: pd.Series
    middle: pd.Series
    upper: pd.Series


@dataclass(frozen=True)
class ADXResult:
    """Average Directional Index with its directional components."""

    adx: pd.Series
    plus_di: pd.Series
    minus_di: pd.Series


@dataclass(frozen=True)
class StochasticResult:
    """Stochastic oscillator ``%K`` and ``%D`` lines, both in ``[0, 100]``."""

    k: pd.Series
    d: pd.Series


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


def sma(close: pd.Series, length: int) -> pd.Series:
    """Simple moving average of *close* over *length* bars."""
    close = _as_float(close, length)
    return _series_or_nan(ta.sma(close, length=length, talib=False), close)


def ema(close: pd.Series, length: int) -> pd.Series:
    """Exponential moving average seeded with the SMA of the first *length* bars.

    The first ``length - 1`` values are NaN, as in TA-Lib.
    """
    close = _as_float(close, length)
    return _series_or_nan(
        ta.ema(close, length=length, talib=False, presma=True),
        close,
    )


def rsi(close: pd.Series, length: int = 14) -> pd.Series:
    """Relative Strength Index with Wilder smoothing, in ``[0, 100]``."""
    close = _as_float(close, length)
    return _series_or_nan(ta.rsi(close, length=length, talib=False), close)


def macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> MACDResult:
    """MACD of *close* with EMA periods *fast*, *slow* and *signal*."""
    close = _as_float(close, fast, slow, signal)
    if fast >= slow:
        msg = f"fast period must be shorter than slow period, got {fast} >= {slow}"
        raise ValueError(msg)
    frame = ta.macd(close, fast=fast, slow=slow, signal=signal, talib=False)
    if frame is None:
        nan = _nan_like(close)
        return MACDResult(macd=nan, signal=nan, histogram=nan)
    return MACDResult(
        macd=indicator_column(frame, "MACD_"),
        signal=indicator_column(frame, "MACDs_"),
        histogram=indicator_column(frame, "MACDh_"),
    )


def bollinger_bands(
    close: pd.Series,
    length: int = 20,
    std: float = 2.0,
    ddof: int = 0,
) -> BollingerBands:
    """Bollinger Bands: SMA(*length*) plus and minus *std* standard deviations.

    ``ddof=0`` (population standard deviation) matches TA-Lib and John
    Bollinger's definition; pandas-ta defaults to the sample deviation.
    """
    close = _as_float(close, length)
    if not 0 <= ddof < length:
        msg = f"ddof must satisfy 0 <= ddof < length, got ddof={ddof}, length={length}"
        raise ValueError(msg)
    frame = ta.bbands(
        close,
        length=length,
        lower_std=std,
        upper_std=std,
        ddof=ddof,
        talib=False,
    )
    if frame is None:
        nan = _nan_like(close)
        return BollingerBands(lower=nan, middle=nan, upper=nan)
    return BollingerBands(
        lower=indicator_column(frame, "BBL_"),
        middle=indicator_column(frame, "BBM_"),
        upper=indicator_column(frame, "BBU_"),
    )


def atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    length: int = 14,
) -> pd.Series:
    """Average True Range with Wilder smoothing, in price units."""
    high, low, close = _as_float_ohlc(high, low, close, length)
    result = ta.atr(high, low, close, length=length, talib=False)
    return _series_or_nan(result, close)


def adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    length: int = 14,
) -> ADXResult:
    """Average Directional Index with ``+DI`` and ``-DI``, each in ``[0, 100]``."""
    high, low, close = _as_float_ohlc(high, low, close, length)
    frame = ta.adx(high, low, close, length=length, talib=False)
    if frame is None:
        nan = _nan_like(close)
        return ADXResult(adx=nan, plus_di=nan, minus_di=nan)
    return ADXResult(
        adx=indicator_column(frame, "ADX_"),
        plus_di=indicator_column(frame, "DMP_"),
        minus_di=indicator_column(frame, "DMN_"),
    )


def stochastic(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k: int = 14,
    d: int = 3,
    smooth_k: int = 3,
) -> StochasticResult:
    """Stochastic oscillator: ``%K`` over *k* bars smoothed by *smooth_k*, ``%D``."""
    high, low, close = _as_float_ohlc(high, low, close, k, d, smooth_k)
    frame = ta.stoch(high, low, close, k=k, d=d, smooth_k=smooth_k, talib=False)
    if frame is None:
        nan = _nan_like(close)
        return StochasticResult(k=nan, d=nan)
    return StochasticResult(
        k=indicator_column(frame, "STOCHk_"),
        d=indicator_column(frame, "STOCHd_"),
    )


def _as_float(series: pd.Series, *lengths: int) -> pd.Series:
    for length in lengths:
        if length < 1:
            msg = f"indicator length must be a positive integer, got {length}"
            raise ValueError(msg)
    return series.astype("float64")


def _as_float_ohlc(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    *lengths: int,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    return _as_float(high, *lengths), _as_float(low), _as_float(close)


def _nan_like(series: pd.Series) -> pd.Series:
    return pd.Series(np.nan, index=series.index, dtype="float64")


def _series_or_nan(result: pd.Series | None, reference: pd.Series) -> pd.Series:
    if result is None:
        return _nan_like(reference)
    return result.astype("float64")
