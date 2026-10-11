"""Named technical indicators backed by TA-Lib.

Outputs preserve the input index and use float64. Warm-up, initialization and
missing-value handling follow TA-Lib: leading NaNs delay the first observation,
and an interior NaN may propagate through the remainder of an indicator. Values
are never filled or computed using future observations. Infinite inputs are
rejected. OHLC inputs must share the same index because TA-Lib works by position.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import talib

if TYPE_CHECKING:
    from numpy.typing import NDArray


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
    """Average Directional Index with its directional indicators."""

    adx: pd.Series
    plus_di: pd.Series
    minus_di: pd.Series


@dataclass(frozen=True)
class StochasticResult:
    """Stochastic oscillator slow ``%K`` and ``%D`` lines."""

    k: pd.Series
    d: pd.Series


def sma(close: pd.Series, length: int) -> pd.Series:
    """Simple moving average; the first ``length - 1`` values are NaN."""
    _validate_period("length", length)
    values = _as_float(close)
    if close.empty:
        return _empty_like(close)
    return _series(talib.SMA(values, timeperiod=length), close)


def ema(close: pd.Series, length: int) -> pd.Series:
    """EMA seeded with the first ``length`` observations' simple average."""
    _validate_period("length", length)
    values = _as_float(close)
    if close.empty:
        return _empty_like(close)
    return _series(talib.EMA(values, timeperiod=length), close)


def rsi(close: pd.Series, length: int = 14) -> pd.Series:
    """Wilder RSI, first defined after ``length`` price changes."""
    _validate_period("length", length)
    values = _as_float(close)
    if close.empty:
        return _empty_like(close)
    return _series(talib.RSI(values, timeperiod=length), close)


def macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> MACDResult:
    """TA-Lib MACD with all outputs warming up for ``slow + signal - 2`` bars.

    TA-Lib seeds the fast EMA from the tail of the slow EMA's initial window.
    Its MACD line therefore differs from subtracting two standalone EMAs.
    """
    _validate_period("fast", fast)
    _validate_period("slow", slow)
    _validate_period("signal", signal, minimum=1)
    if fast >= slow:
        msg = f"fast period must be shorter than slow period, got {fast} >= {slow}"
        raise ValueError(msg)
    values = _as_float(close)
    if close.empty:
        return MACDResult(*(_empty_like(close) for _ in range(3)))
    line, signal_line, histogram = talib.MACD(
        values, fastperiod=fast, slowperiod=slow, signalperiod=signal
    )
    return MACDResult(
        macd=_series(line, close),
        signal=_series(signal_line, close),
        histogram=_series(histogram, close),
    )


def bollinger_bands(
    close: pd.Series,
    length: int = 20,
    std: float = 2.0,
) -> BollingerBands:
    """SMA bands using population standard deviation (``ddof=0``)."""
    _validate_period("length", length)
    if not np.isfinite(std) or not 0 <= std <= 3e37:
        msg = f"std must be finite and between 0 and 3e37, got {std}"
        raise ValueError(msg)
    values = _as_float(close)
    if close.empty:
        return BollingerBands(*(_empty_like(close) for _ in range(3)))
    upper, middle, lower = talib.BBANDS(
        values,
        timeperiod=length,
        nbdevup=std,
        nbdevdn=std,
    )
    return BollingerBands(
        lower=_series(lower, close),
        middle=_series(middle, close),
        upper=_series(upper, close),
    )


def atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    length: int = 14,
) -> pd.Series:
    """Wilder ATR; the first ``length`` values are NaN, including the first bar."""
    _validate_period("length", length, minimum=1)
    high_values, low_values, close_values = _as_float_ohlc(high, low, close)
    if close.empty:
        return _empty_like(close)
    return _series(
        talib.ATR(high_values, low_values, close_values, timeperiod=length), close
    )


def adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    length: int = 14,
) -> ADXResult:
    """Wilder ADX and ``+DI``/``-DI`` with their respective TA-Lib warm-ups.

    ADX starts at position ``2 * length - 1``; DI starts at ``length``.
    """
    _validate_period("length", length)
    high_values, low_values, close_values = _as_float_ohlc(high, low, close)
    if close.empty:
        return ADXResult(*(_empty_like(close) for _ in range(3)))
    return ADXResult(
        adx=_series(
            talib.ADX(high_values, low_values, close_values, timeperiod=length),
            close,
        ),
        plus_di=_series(
            talib.PLUS_DI(high_values, low_values, close_values, timeperiod=length),
            close,
        ),
        minus_di=_series(
            talib.MINUS_DI(high_values, low_values, close_values, timeperiod=length),
            close,
        ),
    )


def stochastic(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k: int = 14,
    d: int = 3,
    smooth_k: int = 3,
) -> StochasticResult:
    """SMA-smoothed stochastic with a shared ``k + smooth_k + d - 3`` warm-up.

    ``smooth_k=1`` leaves fast %K unsmoothed; ``d=1`` makes %D equal %K.
    Both lines are exposed only once %D is available.
    """
    _validate_period("k", k, minimum=1)
    _validate_period("d", d, minimum=1)
    _validate_period("smooth_k", smooth_k, minimum=1)
    high_values, low_values, close_values = _as_float_ohlc(high, low, close)
    if close.empty:
        return StochasticResult(_empty_like(close), _empty_like(close))
    slow_k, slow_d = talib.STOCH(
        high_values,
        low_values,
        close_values,
        fastk_period=k,
        slowk_period=smooth_k,
        slowd_period=d,
    )
    return StochasticResult(k=_series(slow_k, close), d=_series(slow_d, close))


def _validate_period(name: str, value: int, *, minimum: int = 2) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, Integral)
        or not minimum <= value <= 100000
    ):
        msg = f"{name} must be an integer between {minimum} and 100000, got {value}"
        raise ValueError(msg)


def _as_float(series: pd.Series) -> NDArray[np.float64]:
    values = series.to_numpy(dtype=np.float64, copy=True)
    if np.isinf(values).any():
        msg = "indicator inputs must not contain infinity"
        raise ValueError(msg)
    return values


def _as_float_ohlc(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    if not high.index.equals(close.index) or not low.index.equals(close.index):
        msg = "high, low and close must have identical indexes"
        raise ValueError(msg)
    return _as_float(high), _as_float(low), _as_float(close)


def _series(values: NDArray[np.float64], reference: pd.Series) -> pd.Series:
    return pd.Series(values, index=reference.index, dtype="float64")


def _empty_like(series: pd.Series) -> pd.Series:
    return pd.Series(index=series.index, dtype="float64")
