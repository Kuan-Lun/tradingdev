"""Tests for the pandas-ta backed technical feature helpers."""

import numpy as np
import pandas as pd
import pandas_ta as ta
import pytest

from tradingdev.domain.ml.features.direction_features import DirectionFeatureEngineer
from tradingdev.domain.ml.features.features import FeatureEngineer
from tradingdev.domain.ml.features.risk_features import RiskFeatureEngineer
from tradingdev.domain.ml.features.technical_features import (
    bollinger_bands,
    compute_ta_indicators,
    indicator_column,
    macd_histogram,
)


def _named(frame: pd.DataFrame, prefix: str) -> pd.Series:
    """Select a pandas-ta column by prefix, independently of the code under test."""
    names = [str(column) for column in frame.columns if str(column).startswith(prefix)]
    assert len(names) == 1, names
    return frame[names[0]]


def _short_series() -> pd.Series:
    """Too short for pandas-ta to compute MACD, Bollinger Bands or RSI."""
    return pd.Series(np.linspace(100.0, 110.0, 10))


class TestIndicatorColumn:
    def test_selects_unique_prefix(self) -> None:
        frame = pd.DataFrame({"BBL_20_2.0": [1.0], "BBU_20_2.0": [2.0]})
        assert indicator_column(frame, "BBU_").iloc[0] == 2.0

    def test_missing_prefix_raises(self) -> None:
        frame = pd.DataFrame({"BBL_20_2.0": [1.0]})
        with pytest.raises(KeyError, match="BBU_"):
            indicator_column(frame, "BBU_")

    def test_ambiguous_prefix_raises(self) -> None:
        frame = pd.DataFrame({"BB_1": [1.0], "BB_2": [2.0]})
        with pytest.raises(KeyError, match="BB_"):
            indicator_column(frame, "BB_")


class TestMacdHistogram:
    def test_returns_histogram_not_signal_line(
        self, large_ohlcv_df: pd.DataFrame
    ) -> None:
        close = large_ohlcv_df["close"].astype(float)
        hist = macd_histogram(close)
        assert hist is not None
        raw = ta.macd(close)
        pd.testing.assert_series_equal(hist, _named(raw, "MACDh_"), check_names=False)
        assert not np.allclose(hist.dropna(), _named(raw, "MACDs_").dropna())

    def test_short_series_returns_none(self) -> None:
        assert macd_histogram(_short_series()) is None


class TestBollingerBands:
    def test_bands_are_named_and_ordered(self, large_ohlcv_df: pd.DataFrame) -> None:
        close = large_ohlcv_df["close"].astype(float)
        bands = bollinger_bands(close, length=20)
        assert bands is not None
        lower, middle, upper = bands
        raw = ta.bbands(close, length=20)
        pd.testing.assert_series_equal(lower, _named(raw, "BBL_"), check_names=False)
        pd.testing.assert_series_equal(middle, _named(raw, "BBM_"), check_names=False)
        pd.testing.assert_series_equal(upper, _named(raw, "BBU_"), check_names=False)
        valid = lower.notna()
        assert (lower[valid] <= middle[valid]).all()
        assert (middle[valid] <= upper[valid]).all()

    def test_short_series_returns_none(self) -> None:
        assert bollinger_bands(_short_series(), length=20) is None


class TestComputeTaIndicators:
    def test_bb_pctb_matches_pandas_ta_percent_b(
        self, large_ohlcv_df: pd.DataFrame
    ) -> None:
        close = large_ohlcv_df["close"].astype(float)
        features = compute_ta_indicators(close)
        expected = _named(ta.bbands(close, length=20), "BBP_")
        pd.testing.assert_series_equal(features["bb_pctb"], expected, check_names=False)

    def test_bb_pctb_above_half_when_close_above_middle_band(
        self, large_ohlcv_df: pd.DataFrame
    ) -> None:
        """Regression: positional band selection produced ``1 - %B``."""
        close = large_ohlcv_df["close"].astype(float)
        bands = bollinger_bands(close, length=20)
        assert bands is not None
        _, middle, _ = bands
        pctb = compute_ta_indicators(close)["bb_pctb"]
        above = (close > middle) & pctb.notna()
        assert above.any()
        assert (pctb[above] > 0.5).all()

    def test_macd_hist_matches_pandas_ta_histogram(
        self, large_ohlcv_df: pd.DataFrame
    ) -> None:
        close = large_ohlcv_df["close"].astype(float)
        features = compute_ta_indicators(close)
        expected = _named(ta.macd(close), "MACDh_")
        pd.testing.assert_series_equal(
            features["macd_hist"], expected, check_names=False
        )

    def test_short_series_yields_no_indicators(self) -> None:
        assert compute_ta_indicators(_short_series()) == {}


def _expected_by_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    close = df["close"].astype(float)
    return pd.DataFrame(
        {
            "timestamp": df["timestamp"],
            "expected_pctb": _named(ta.bbands(close, length=20), "BBP_").to_numpy(),
            "expected_hist": _named(ta.macd(close), "MACDh_").to_numpy(),
        }
    )


class TestFeatureEngineersUseNamedColumns:
    @pytest.mark.parametrize(
        "engineer",
        [
            FeatureEngineer(lookback=24),
            DirectionFeatureEngineer(lookback=60, prediction_horizon=5),
        ],
        ids=["FeatureEngineer", "DirectionFeatureEngineer"],
    )
    def test_bb_pctb_and_macd_hist_semantics(
        self,
        engineer: FeatureEngineer | DirectionFeatureEngineer,
        large_ohlcv_df: pd.DataFrame,
    ) -> None:
        result = engineer.transform(large_ohlcv_df, include_target=False)
        joined = result[["timestamp", "bb_pctb", "macd_hist"]].merge(
            _expected_by_timestamp(large_ohlcv_df), on="timestamp"
        )
        assert len(joined) == len(result)
        np.testing.assert_allclose(joined["bb_pctb"], joined["expected_pctb"])
        np.testing.assert_allclose(joined["macd_hist"], joined["expected_hist"])

    def test_risk_features_bollinger_width_is_positive(
        self, large_ohlcv_df: pd.DataFrame
    ) -> None:
        """Regression: swapped bands produced a negative bandwidth."""
        result = RiskFeatureEngineer(lookback=24).transform(
            large_ohlcv_df, include_target=False
        )
        assert (result["bb_width_20"] > 0).all()
