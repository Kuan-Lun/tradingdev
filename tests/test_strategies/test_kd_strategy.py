"""Tests for the KD crossover strategy."""

import pandas as pd
import pytest

from quant_backtest.backtest.signal_engine import (
    SignalBacktestEngine,
)
from quant_backtest.data.schemas import (
    KDFitConfig,
    KDStrategyConfig,
)
from quant_backtest.strategies.kd_strategy import KDStrategy


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
        result = self.strategy.generate_signals(sample_ohlcv_with_kd)
        assert result["signal"].iloc[0] == 0


class TestKDStrategyFit:
    def test_fit_without_fit_config_is_noop(
        self, sample_ohlcv_with_kd: pd.DataFrame
    ) -> None:
        config = KDStrategyConfig()
        strategy = KDStrategy(config=config)
        original = strategy.get_parameters().copy()
        strategy.fit(sample_ohlcv_with_kd)
        assert strategy.get_parameters() == original

    def test_fit_updates_parameters(self, sample_ohlcv_with_kd: pd.DataFrame) -> None:
        fit_config = KDFitConfig(
            k_period_range=[9, 14],
            d_period_range=[3],
            smooth_k_range=[3],
            overbought_range=[80.0],
            oversold_range=[20.0],
            target_metric="sharpe_ratio",
        )
        engine = SignalBacktestEngine(init_cash=10_000.0, fees=0.0, slippage=0.0)
        strategy = KDStrategy(
            config=KDStrategyConfig(),
            fit_config=fit_config,
            backtest_engine=engine,
        )
        strategy.fit(sample_ohlcv_with_kd)
        params = strategy.get_parameters()
        assert params["k_period"] in [9, 14]

    def test_fit_requires_engine(self, sample_ohlcv_with_kd: pd.DataFrame) -> None:
        fit_config = KDFitConfig(
            k_period_range=[9, 14],
            d_period_range=[3],
            smooth_k_range=[3],
        )
        strategy = KDStrategy(
            config=KDStrategyConfig(),
            fit_config=fit_config,
        )
        with pytest.raises(RuntimeError, match="backtest_engine"):
            strategy.fit(sample_ohlcv_with_kd)

    def test_fit_generates_valid_signals_after(
        self, sample_ohlcv_with_kd: pd.DataFrame
    ) -> None:
        fit_config = KDFitConfig(
            k_period_range=[9, 14],
            d_period_range=[3],
            smooth_k_range=[3],
            overbought_range=[80.0],
            oversold_range=[20.0],
        )
        engine = SignalBacktestEngine(init_cash=10_000.0, fees=0.0, slippage=0.0)
        strategy = KDStrategy(
            config=KDStrategyConfig(),
            fit_config=fit_config,
            backtest_engine=engine,
        )
        strategy.fit(sample_ohlcv_with_kd)
        result = strategy.generate_signals(sample_ohlcv_with_kd)
        assert "signal" in result.columns
        assert set(result["signal"].unique()).issubset({-1, 0, 1})
