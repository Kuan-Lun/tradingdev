"""Numerical and input contracts for the TA-Lib indicator layer."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest
import talib

from tradingdev.domain import indicators

OHLC = tuple[pd.Series, pd.Series, pd.Series]
Compute = Callable[[pd.Series, pd.Series, pd.Series], list[pd.Series]]


@pytest.fixture
def ohlc(sample_ohlcv_with_kd: pd.DataFrame) -> OHLC:
    df = sample_ohlcv_with_kd
    return df["high"], df["low"], df["close"]


def _assert_close(actual: pd.Series, expected: pd.Series) -> None:
    assert actual.index.equals(expected.index)
    np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-10, equal_nan=True)


# Expected warm-up positions and values for a completely flat OHLC series.
_INDICATORS: list[tuple[str, Compute, tuple[int, ...], tuple[float, ...]]] = [
    ("sma", lambda _h, _lo, c: [indicators.sma(c, 20)], (19,), (100.0,)),
    ("ema", lambda _h, _lo, c: [indicators.ema(c, 21)], (20,), (100.0,)),
    ("rsi", lambda _h, _lo, c: [indicators.rsi(c)], (14,), (0.0,)),
    (
        "macd",
        lambda _h, _lo, c: [
            indicators.macd(c).macd,
            indicators.macd(c).signal,
            indicators.macd(c).histogram,
        ],
        (33, 33, 33),
        (0.0, 0.0, 0.0),
    ),
    (
        "bollinger_bands",
        lambda _h, _lo, c: [
            indicators.bollinger_bands(c).lower,
            indicators.bollinger_bands(c).middle,
            indicators.bollinger_bands(c).upper,
        ],
        (19, 19, 19),
        (100.0, 100.0, 100.0),
    ),
    ("atr", lambda h, lo, c: [indicators.atr(h, lo, c)], (14,), (0.0,)),
    (
        "adx",
        lambda h, lo, c: [
            indicators.adx(h, lo, c).adx,
            indicators.adx(h, lo, c).plus_di,
            indicators.adx(h, lo, c).minus_di,
        ],
        (27, 14, 14),
        (0.0, 0.0, 0.0),
    ),
    (
        "stochastic",
        lambda h, lo, c: [
            indicators.stochastic(h, lo, c).k,
            indicators.stochastic(h, lo, c).d,
        ],
        (17, 17),
        (0.0, 0.0),
    ),
]
_IDS = [name for name, *_ in _INDICATORS]
_COMPUTATIONS = [(name, compute) for name, compute, *_ in _INDICATORS]


def _ema_reference(close: pd.Series, length: int, start: int = 0) -> pd.Series:
    """Direct recurrence with an explicit seed window, independent of TA-Lib."""
    expected = pd.Series(np.nan, index=close.index, dtype="float64")
    first = start + length - 1
    expected.iloc[first] = close.iloc[start : first + 1].mean()
    for position in range(first + 1, len(close)):
        previous = expected.iloc[position - 1]
        expected.iloc[position] = previous + 2 / (length + 1) * (
            close.iloc[position] - previous
        )
    return expected


class TestNumericalDefinitions:
    def test_sma_matches_rolling_mean(self, ohlc: OHLC) -> None:
        close = ohlc[2]
        _assert_close(indicators.sma(close, 20), close.rolling(20).mean())

    def test_ema_uses_sma_seed_and_recursive_updates(self, ohlc: OHLC) -> None:
        close = ohlc[2]
        _assert_close(indicators.ema(close, 21), _ema_reference(close, 21))

    def test_rsi_uses_wilder_seed_and_updates(self) -> None:
        close = pd.Series([10.0, 12.0, 11.0, 14.0, 13.0, 15.0])
        expected = pd.Series([np.nan, np.nan, np.nan, 100 * 5 / 6, 200 / 3, 475 / 6])
        _assert_close(indicators.rsi(close, length=3), expected)

    def test_atr_excludes_first_bar_and_uses_wilder_updates(self) -> None:
        close = pd.Series([10.0, 12.0, 11.0, 14.0, 13.0, 15.0])
        high = close + pd.Series([1, 1, 2, 2, 1, 2])
        low = close - pd.Series([2, 1, 1, 1, 2, 1])
        expected = pd.Series([np.nan, np.nan, np.nan, 11 / 3, 31 / 9, 98 / 27])
        _assert_close(indicators.atr(high, low, close, length=3), expected)
        _assert_close(
            indicators.atr(high, low, close, length=1),
            pd.Series([np.nan, 3.0, 3.0, 5.0, 3.0, 4.0]),
        )

    def test_macd_aligns_seed_windows_and_output_warmup(self, ohlc: OHLC) -> None:
        close = ohlc[2]
        expected_line = _ema_reference(close, 3, start=4) - _ema_reference(close, 7)
        expected_signal = _ema_reference(expected_line, 4, start=6)
        expected_line = expected_line.where(expected_signal.notna())
        result = indicators.macd(close, fast=3, slow=7, signal=4)
        _assert_close(result.macd, expected_line)
        _assert_close(result.signal, expected_signal)
        _assert_close(result.histogram, expected_line - expected_signal)
        standalone = _ema_reference(close, 3) - _ema_reference(close, 7)
        assert result.macd.iloc[9] != pytest.approx(standalone.iloc[9])

    def test_macd_signal_one_has_no_extra_smoothing(self, ohlc: OHLC) -> None:
        result = indicators.macd(ohlc[2], fast=3, slow=7, signal=1)
        assert result.macd.first_valid_index() == ohlc[2].index[6]
        _assert_close(result.signal, result.macd)
        _assert_close(result.histogram, result.macd * 0)

    @pytest.mark.parametrize("std", [0.0, 1.5, 2.0])
    def test_bollinger_bands_use_population_std(self, std: float, ohlc: OHLC) -> None:
        close = ohlc[2]
        bands = indicators.bollinger_bands(close, length=20, std=std)
        mean = close.rolling(20).mean()
        deviation = std * close.rolling(20).std(ddof=0)
        _assert_close(bands.middle, mean)
        _assert_close(bands.upper, mean + deviation)
        _assert_close(bands.lower, mean - deviation)

    @pytest.mark.parametrize(
        ("k", "d", "smooth_k"), [(5, 2, 4), (5, 1, 3), (5, 3, 1), (1, 1, 1)]
    )
    def test_stochastic_smoothing_and_shared_warmup(
        self, k: int, d: int, smooth_k: int, ohlc: OHLC
    ) -> None:
        high, low, close = ohlc
        lowest = low.rolling(k).min()
        highest = high.rolling(k).max()
        raw_k = 100 * (close - lowest) / (highest - lowest)
        expected_k = raw_k.rolling(smooth_k).mean()
        expected_d = expected_k.rolling(d).mean()
        expected_k = expected_k.where(expected_d.notna())
        result = indicators.stochastic(high, low, close, k=k, d=d, smooth_k=smooth_k)
        _assert_close(result.k, expected_k)
        _assert_close(result.d, expected_d)

    def test_adx_returns_directional_indicators_not_directional_movement(
        self, ohlc: OHLC
    ) -> None:
        high, low, close = ohlc
        result = indicators.adx(high, low, close, length=7)
        high_values, low_values, close_values = (
            series.to_numpy(dtype=np.float64) for series in ohlc
        )
        expected = (
            talib.ADX(high_values, low_values, close_values, timeperiod=7),
            talib.PLUS_DI(high_values, low_values, close_values, timeperiod=7),
            talib.MINUS_DI(high_values, low_values, close_values, timeperiod=7),
        )
        for actual, values in zip(
            (result.adx, result.plus_di, result.minus_di), expected, strict=True
        ):
            _assert_close(actual, pd.Series(values, index=close.index))
        assert not np.allclose(
            result.plus_di.iloc[7:],
            talib.PLUS_DM(high_values, low_values, timeperiod=7)[7:],
        )


class TestShapeAndWarmup:
    @pytest.mark.parametrize(
        ("name", "compute", "warmups", "flat_values"), _INDICATORS, ids=_IDS
    )
    def test_exact_warmup_and_flat_market(
        self,
        name: str,
        compute: Compute,
        warmups: tuple[int, ...],
        flat_values: tuple[float, ...],
    ) -> None:
        close = pd.Series(100.0, index=pd.RangeIndex(80))
        for output, warmup, value in zip(
            compute(close, close, close), warmups, flat_values, strict=True
        ):
            expected = pd.Series(value, index=close.index)
            expected.iloc[:warmup] = np.nan
            _assert_close(output, expected)

    @pytest.mark.parametrize(("name", "compute"), _COMPUTATIONS, ids=_IDS)
    @pytest.mark.parametrize("size", [0, 1, 5])
    def test_empty_and_short_inputs_preserve_shape(
        self, name: str, compute: Compute, size: int
    ) -> None:
        index = pd.date_range("2024-01-01", periods=size, freq="h", tz="UTC")
        close = pd.Series(np.arange(size), index=index, dtype="int64")
        for series in compute(close + 1, close - 1, close):
            assert series.index.equals(index), name
            assert series.dtype == np.float64, name
            assert series.isna().all(), name

    @pytest.mark.parametrize(("name", "compute"), _COMPUTATIONS, ids=_IDS)
    def test_preserves_datetime_index_dtype_and_inputs(
        self, name: str, compute: Compute
    ) -> None:
        index = pd.date_range("2024-01-01", periods=120, freq="h", tz="UTC")
        close = pd.Series(np.arange(120), index=index, dtype="int64")
        inputs = (close + 2, close - 1, close)
        originals = tuple(series.copy(deep=True) for series in inputs)
        for output in compute(*inputs):
            assert output.index.equals(index), name
            assert output.dtype == np.float64, name
        for actual, expected in zip(inputs, originals, strict=True):
            pd.testing.assert_series_equal(actual, expected)

    @pytest.mark.parametrize(("name", "compute"), _COMPUTATIONS, ids=_IDS)
    def test_outputs_depend_only_on_available_prefix(
        self, name: str, compute: Compute, ohlc: OHLC
    ) -> None:
        complete = compute(*ohlc)
        for size in (6, 17, 20, 28, 33, 34, 80):
            prefix = compute(*(series.iloc[:size] for series in ohlc))
            for actual, expected in zip(prefix, complete, strict=True):
                _assert_close(actual, expected.iloc[:size])


class TestMissingValues:
    @pytest.mark.parametrize(("name", "compute"), _COMPUTATIONS, ids=_IDS)
    def test_leading_nan_delays_warmup_without_filling(
        self, name: str, compute: Compute, ohlc: OHLC
    ) -> None:
        inputs = tuple(series.copy() for series in ohlc)
        for series in inputs:
            series.iloc[:5] = np.nan
        actual = compute(*inputs)
        expected = compute(*(series.iloc[5:] for series in ohlc))
        for result, reference in zip(actual, expected, strict=True):
            _assert_close(result, reference.reindex(ohlc[2].index))

    @pytest.mark.parametrize(("name", "compute"), _COMPUTATIONS, ids=_IDS)
    def test_all_nan_preserves_index(self, name: str, compute: Compute) -> None:
        close = pd.Series(np.nan, index=pd.RangeIndex(120))
        for output in compute(close, close, close):
            _assert_close(output, close)

    def test_interior_nan_follows_native_backend(self, ohlc: OHLC) -> None:
        high, low, close = (series.copy() for series in ohlc)
        close.iloc[50] = np.nan
        high_values, low_values, close_values = (
            series.to_numpy(dtype=np.float64) for series in (high, low, close)
        )
        line, signal, histogram = talib.MACD(close_values)
        upper, middle, lower = talib.BBANDS(close_values, timeperiod=20)
        slow_k, slow_d = talib.STOCH(
            high_values, low_values, close_values, fastk_period=14
        )
        expected = {
            "sma": [talib.SMA(close_values, timeperiod=20)],
            "ema": [talib.EMA(close_values, timeperiod=21)],
            "rsi": [talib.RSI(close_values)],
            "macd": [line, signal, histogram],
            "bollinger_bands": [lower, middle, upper],
            "atr": [talib.ATR(high_values, low_values, close_values)],
            "adx": [
                talib.ADX(high_values, low_values, close_values),
                talib.PLUS_DI(high_values, low_values, close_values),
                talib.MINUS_DI(high_values, low_values, close_values),
            ],
            "stochastic": [slow_k, slow_d],
        }
        for name, compute in _COMPUTATIONS:
            for actual, reference in zip(
                compute(high, low, close), expected[name], strict=True
            ):
                _assert_close(actual, pd.Series(reference, index=close.index))
        # Unlike a pandas rolling mean, TA-Lib SMA does not restart after the gap.
        assert indicators.sma(close, 20).iloc[50:].isna().all()
        assert close.rolling(20).mean().iloc[70:].notna().all()


class TestValidation:
    @pytest.mark.parametrize("length", [-1, 0, 1, 100001, True, 2.5])
    def test_invalid_ma_period_is_rejected(self, length: int, ohlc: OHLC) -> None:
        with pytest.raises(ValueError, match="length.*integer between 2 and 100000"):
            indicators.sma(ohlc[2], length)

    @pytest.mark.parametrize(
        "call",
        [
            lambda h, lo, c: indicators.ema(c, 1),
            lambda h, lo, c: indicators.rsi(c, 1),
            lambda h, lo, c: indicators.bollinger_bands(c, 1),
            lambda h, lo, c: indicators.adx(h, lo, c, 1),
            lambda h, lo, c: indicators.atr(h, lo, c, 0),
            lambda h, lo, c: indicators.macd(c, fast=1),
            lambda h, lo, c: indicators.macd(c, signal=0),
            lambda h, lo, c: indicators.stochastic(h, lo, c, k=0),
            lambda h, lo, c: indicators.stochastic(h, lo, c, d=0),
            lambda h, lo, c: indicators.stochastic(h, lo, c, smooth_k=0),
        ],
    )
    def test_indicator_specific_period_ranges(
        self, call: Callable[[pd.Series, pd.Series, pd.Series], object], ohlc: OHLC
    ) -> None:
        with pytest.raises(ValueError, match="integer between"):
            call(*ohlc)

    @pytest.mark.parametrize(("fast", "slow"), [(26, 12), (12, 12)])
    def test_macd_fast_must_be_shorter_than_slow(
        self, fast: int, slow: int, ohlc: OHLC
    ) -> None:
        with pytest.raises(ValueError, match="fast period must be shorter"):
            indicators.macd(ohlc[2], fast=fast, slow=slow)

    @pytest.mark.parametrize("std", [-1.0, np.nan, np.inf, -np.inf, 4e37])
    def test_invalid_bollinger_deviation_is_rejected(
        self, std: float, ohlc: OHLC
    ) -> None:
        with pytest.raises(ValueError, match="std must be finite"):
            indicators.bollinger_bands(ohlc[2], std=std)

    @pytest.mark.parametrize(
        "call", [indicators.atr, indicators.adx, indicators.stochastic]
    )
    @pytest.mark.parametrize("mismatch", ["order", "length", "labels"])
    def test_ohlc_indexes_must_match(
        self,
        call: Callable[[pd.Series, pd.Series, pd.Series], object],
        mismatch: str,
        ohlc: OHLC,
    ) -> None:
        high, low, close = ohlc
        if mismatch == "order":
            high = high.iloc[::-1]
        elif mismatch == "length":
            low = low.iloc[:-1]
        else:
            low = low.set_axis(low.index + 1)
        with pytest.raises(ValueError, match="identical indexes"):
            call(high, low, close)

    @pytest.mark.parametrize(("name", "compute"), _COMPUTATIONS, ids=_IDS)
    @pytest.mark.parametrize("value", [np.inf, -np.inf])
    def test_infinite_inputs_are_rejected(
        self, name: str, compute: Compute, value: float, ohlc: OHLC
    ) -> None:
        close = ohlc[2].copy()
        close.iloc[30] = value
        with pytest.raises(ValueError, match="infinity"):
            compute(ohlc[0], ohlc[1], close)
