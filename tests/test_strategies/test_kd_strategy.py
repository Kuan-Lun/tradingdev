"""Tests for the KD crossover strategy."""

import pandas as pd

from btc_strategy.data.schemas import KDStrategyConfig
from btc_strategy.strategies.kd_strategy import KDStrategy


class TestKDStrategy:
    def setup_method(self) -> None:
        self.config = KDStrategyConfig()
        self.strategy = KDStrategy(config=self.config)

    def test_signal_column_exists(self, sample_ohlcv_with_kd: pd.DataFrame) -> None:
        result = self.strategy.generate_signals(sample_ohlcv_with_kd)
        assert "signal" in result.columns

    def test_signal_values_valid(self, sample_ohlcv_with_kd: pd.DataFrame) -> None:
        result = self.strategy.generate_signals(sample_ohlcv_with_kd)
        unique_signals = set(result["signal"].unique())
        assert unique_signals.issubset({-1, 0, 1})

    def test_indicator_columns_present(
        self, sample_ohlcv_with_kd: pd.DataFrame
    ) -> None:
        result = self.strategy.generate_signals(sample_ohlcv_with_kd)
        assert "stoch_k" in result.columns
        assert "stoch_d" in result.columns

    def test_get_parameters(self) -> None:
        params = self.strategy.get_parameters()
        assert params["k_period"] == 14
        assert params["overbought"] == 80.0
        assert params["oversold"] == 20.0

    def test_custom_config(self, sample_ohlcv_with_kd: pd.DataFrame) -> None:
        config = KDStrategyConfig(overbought=90.0, oversold=10.0)
        strategy = KDStrategy(config=config)
        result = strategy.generate_signals(sample_ohlcv_with_kd)
        assert "signal" in result.columns

    def test_no_look_ahead_bias(self, sample_ohlcv_with_kd: pd.DataFrame) -> None:
        """First bar always has signal=0 (no prior data for crossover)."""
        result = self.strategy.generate_signals(sample_ohlcv_with_kd)
        assert result["signal"].iloc[0] == 0
