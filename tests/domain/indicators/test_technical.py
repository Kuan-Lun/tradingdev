"""Tests for the pandas-ta backed indicator layer."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
import pandas_ta
import pytest

from tradingdev.domain import indicators

OHLC = tuple[pd.Series, pd.Series, pd.Series]
Compute = Callable[[pd.Series, pd.Series, pd.Series], list[pd.Series]]


@pytest.fixture
def ohlc(sample_ohlcv_with_kd: pd.DataFrame) -> OHLC:
    df = sample_ohlcv_with_kd
    return df["high"], df["low"], df["close"]


def _assert_close(actual: pd.Series, expected: pd.Series, rtol: float = 1e-9) -> None:
    """Compare two series where both are defined; NaN masks must agree."""
    assert actual.index.equals(expected.index)
    pd.testing.assert_series_equal(actual.isna(), expected.isna(), check_names=False)
    defined = actual.notna()
    assert defined.sum() > 0
    np.testing.assert_allclose(actual[defined], expected[defined], rtol=rtol)


_INDICATORS: list[tuple[str, Compute]] = [
    ("sma", lambda _h, _lo, c: [indicators.sma(c, 20)]),
    ("ema", lambda _h, _lo, c: [indicators.ema(c, 21)]),
    ("rsi", lambda _h, _lo, c: [indicators.rsi(c)]),
    (
        "macd",
        lambda _h, _lo, c: [
            indicators.macd(c).macd,
            indicators.macd(c).signal,
            indicators.macd(c).histogram,
        ],
    ),
    (
        "bollinger_bands",
        lambda _h, _lo, c: [
            indicators.bollinger_bands(c).lower,
            indicators.bollinger_bands(c).middle,
            indicators.bollinger_bands(c).upper,
        ],
    ),
    ("atr", lambda h, lo, c: [indicators.atr(h, lo, c)]),
    (
        "adx",
        lambda h, lo, c: [
            indicators.adx(h, lo, c).adx,
            indicators.adx(h, lo, c).plus_di,
            indicators.adx(h, lo, c).minus_di,
        ],
    ),
    (
        "stochastic",
        lambda h, lo, c: [
            indicators.stochastic(h, lo, c).k,
            indicators.stochastic(h, lo, c).d,
        ],
    ),
]
_IDS = [name for name, _ in _INDICATORS]


class TestMovingAverages:
    def test_sma_matches_rolling_mean(self, ohlc: OHLC) -> None:
        close = ohlc[2]
        _assert_close(indicators.sma(close, 20), close.rolling(20).mean())

    def test_ema_is_sma_seeded_and_converges_to_pandas_ewm(self, ohlc: OHLC) -> None:
        close = ohlc[2]
        result = indicators.ema(close, 21)
        assert result.iloc[:20].isna().all()
        assert result.iloc[20] == pytest.approx(close.iloc[:21].mean())
        pandas_ewm = close.ewm(span=21, adjust=False).mean()
        np.testing.assert_allclose(result.iloc[-20:], pandas_ewm.iloc[-20:], rtol=1e-6)


class TestOscillators:
    def test_rsi_is_bounded(self, ohlc: OHLC) -> None:
        result = indicators.rsi(ohlc[2], length=14)
        assert result.dropna().between(0, 100).all()

    def test_stochastic_is_bounded_and_d_smooths_k(self, ohlc: OHLC) -> None:
        stoch = indicators.stochastic(*ohlc, k=14, d=3, smooth_k=3)
        assert stoch.k.dropna().between(0, 100).all()
        assert stoch.d.dropna().between(0, 100).all()
        _assert_close(stoch.d, stoch.k.rolling(3).mean())

    def test_macd_components_are_consistent(self, ohlc: OHLC) -> None:
        close = ohlc[2]
        result = indicators.macd(close, fast=12, slow=26, signal=9)
        _assert_close(result.histogram, result.macd - result.signal)
        defined = result.macd.notna()
        expected_line = indicators.ema(close, 12) - indicators.ema(close, 26)
        np.testing.assert_allclose(result.macd[defined], expected_line[defined])


class TestVolatilityAndTrend:
    def test_bollinger_bands_use_population_std(self, ohlc: OHLC) -> None:
        close = ohlc[2]
        bands = indicators.bollinger_bands(close, length=20, std=2.0)
        _assert_close(bands.middle, close.rolling(20).mean())
        _assert_close(bands.upper - bands.middle, 2.0 * close.rolling(20).std(ddof=0))
        _assert_close(bands.middle - bands.lower, 2.0 * close.rolling(20).std(ddof=0))

    def test_bollinger_ddof_selects_sample_std(self, ohlc: OHLC) -> None:
        close = ohlc[2]
        bands = indicators.bollinger_bands(close, length=20, ddof=1)
        _assert_close(bands.upper - bands.middle, 2.0 * close.rolling(20).std(ddof=1))

    def test_atr_is_non_negative(self, ohlc: OHLC) -> None:
        result = indicators.atr(*ohlc, length=14)
        assert (result.dropna() >= 0).all()

    def test_adx_components_are_bounded(self, ohlc: OHLC) -> None:
        result = indicators.adx(*ohlc, length=14)
        for series in (result.adx, result.plus_di, result.minus_di):
            assert series.dropna().between(0, 100).all()


class TestShapeContract:
    @pytest.mark.parametrize(("name", "compute"), _INDICATORS, ids=_IDS)
    def test_preserves_index_length_and_dtype(
        self, name: str, compute: Compute, ohlc: OHLC
    ) -> None:
        for series in compute(*ohlc):
            assert series.index.equals(ohlc[2].index), name
            assert series.dtype == np.float64, name

    @pytest.mark.parametrize(("name", "compute"), _INDICATORS, ids=_IDS)
    def test_preserves_datetime_index(self, name: str, compute: Compute) -> None:
        index = pd.date_range("2024-01-01", periods=120, freq="h", tz="UTC")
        close = pd.Series(np.linspace(100.0, 130.0, 120), index=index)
        for series in compute(close + 1.0, close - 1.0, close):
            assert series.index.equals(index), name

    @pytest.mark.parametrize(("name", "compute"), _INDICATORS, ids=_IDS)
    def test_short_input_returns_nan_not_none(
        self, name: str, compute: Compute
    ) -> None:
        close = pd.Series(np.linspace(100.0, 110.0, 5))
        for series in compute(close + 1.0, close - 1.0, close):
            assert len(series) == 5, name
            assert series.isna().all(), name


class TestValidation:
    def test_non_positive_length_is_rejected(self, ohlc: OHLC) -> None:
        with pytest.raises(ValueError, match="positive"):
            indicators.sma(ohlc[2], 0)

    def test_macd_fast_must_be_shorter_than_slow(self, ohlc: OHLC) -> None:
        with pytest.raises(ValueError, match="fast"):
            indicators.macd(ohlc[2], fast=26, slow=12)

    def test_bollinger_ddof_out_of_range_is_rejected(self, ohlc: OHLC) -> None:
        with pytest.raises(ValueError, match="ddof"):
            indicators.bollinger_bands(ohlc[2], length=20, ddof=20)


class TestIndicatorColumn:
    def test_selects_unique_prefix(self) -> None:
        frame = pd.DataFrame({"BBL_20_2.0": [1.0], "BBU_20_2.0": [2.0]})
        assert indicators.indicator_column(frame, "BBU_").iloc[0] == 2.0

    def test_missing_prefix_raises(self) -> None:
        frame = pd.DataFrame({"BBL_20_2.0": [1.0]})
        with pytest.raises(KeyError, match="BBU_"):
            indicators.indicator_column(frame, "BBU_")

    def test_ambiguous_prefix_raises(self) -> None:
        frame = pd.DataFrame({"BB_1": [1.0], "BB_2": [2.0]})
        with pytest.raises(KeyError, match="BB_"):
            indicators.indicator_column(frame, "BB_")


class TestTalibDelegationIsPinned:
    """Results must not depend on whether TA-Lib happens to be importable."""

    @pytest.mark.parametrize(
        ("pandas_ta_name", "call"),
        [
            ("sma", lambda _h, _lo, c: indicators.sma(c, 20)),
            ("ema", lambda _h, _lo, c: indicators.ema(c, 21)),
            ("rsi", lambda _h, _lo, c: indicators.rsi(c)),
            ("macd", lambda _h, _lo, c: indicators.macd(c)),
            ("bbands", lambda _h, _lo, c: indicators.bollinger_bands(c)),
            ("atr", lambda h, lo, c: indicators.atr(h, lo, c)),
            ("adx", lambda h, lo, c: indicators.adx(h, lo, c)),
            ("stoch", lambda h, lo, c: indicators.stochastic(h, lo, c)),
        ],
    )
    def test_every_call_disables_delegation(
        self,
        monkeypatch: pytest.MonkeyPatch,
        pandas_ta_name: str,
        call: Callable[[pd.Series, pd.Series, pd.Series], object],
        ohlc: OHLC,
    ) -> None:
        seen: list[object] = []

        def recorder(*_args: object, **kwargs: object) -> None:
            seen.append(kwargs.get("talib"))

        monkeypatch.setattr(pandas_ta, pandas_ta_name, recorder)
        call(*ohlc)
        assert seen == [False]
