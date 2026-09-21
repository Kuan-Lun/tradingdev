"""Tests for the quantile / path-opportunity feature engineer."""

import numpy as np
import pandas as pd

from tradingdev.domain import indicators
from tradingdev.domain.ml.features.quantile_features import QuantileFeatureEngineer


class TestQuantileFeatureEngineer:
    def test_transform_produces_features_without_nan(
        self, large_ohlcv_df: pd.DataFrame
    ) -> None:
        fe = QuantileFeatureEngineer(horizon=30, profit_target=0.001)
        result = fe.transform(large_ohlcv_df, include_target=True, target_type="path")
        names = fe.get_feature_names()
        for column in ("atr_14", "rsi_14", "close_sma_ratio_7", "vol_sma_ratio_7"):
            assert column in names
        assert result[names].isna().sum().sum() == 0
        assert set(result["target_long"].unique()) <= {0.0, 1.0}
        assert set(result["target_short"].unique()) <= {0.0, 1.0}

    def test_atr_and_rsi_come_from_the_indicator_layer(
        self, large_ohlcv_df: pd.DataFrame
    ) -> None:
        result = QuantileFeatureEngineer(horizon=30).transform(
            large_ohlcv_df, include_target=False
        )
        high = large_ohlcv_df["high"].astype(float)
        low = large_ohlcv_df["low"].astype(float)
        close = large_ohlcv_df["close"].astype(float)
        expected = pd.DataFrame(
            {
                "timestamp": large_ohlcv_df["timestamp"],
                "expected_atr": (
                    indicators.atr(high, low, close, length=14) / close
                ).to_numpy(),
                "expected_rsi": indicators.rsi(close, length=14).to_numpy(),
            }
        )
        joined = result[["timestamp", "atr_14", "rsi_14"]].merge(
            expected, on="timestamp"
        )
        assert len(joined) == len(result)
        np.testing.assert_allclose(joined["atr_14"], joined["expected_atr"])
        np.testing.assert_allclose(joined["rsi_14"], joined["expected_rsi"])

    def test_regime_target_uses_four_classes(
        self, large_ohlcv_df: pd.DataFrame
    ) -> None:
        fe = QuantileFeatureEngineer(horizon=10, profit_target=0.001)
        result = fe.transform(large_ohlcv_df, include_target=True, target_type="regime")
        assert set(result["target_regime"].unique()) <= {0.0, 1.0, 2.0, 3.0}
