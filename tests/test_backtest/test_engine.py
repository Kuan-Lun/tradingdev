"""Tests for the backtest engines."""

from datetime import UTC

import numpy as np
import pandas as pd

from btc_strategy.backtest.result import BacktestResult
from btc_strategy.backtest.signal_engine import (
    SignalBacktestEngine,
)
from btc_strategy.backtest.volume_engine import (
    VolumeBacktestEngine,
)


def _make_ohlcv_df(
    prices: list[float],
    signals: list[int],
    spread: float = 0.5,
) -> pd.DataFrame:
    """Build a DataFrame with OHLCV + signal columns."""
    n = len(prices)
    close = np.array(prices)
    return pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2024-01-01", periods=n, freq="h", tz=UTC
            ),
            "open": close - spread * 0.1,
            "high": close + spread,
            "low": close - spread,
            "close": close,
            "volume": [1000.0] * n,
            "signal": signals,
        }
    )


EXPECTED_KEYS = {
    "total_return",
    "total_pnl_usdt",
    "sharpe_ratio",
    "max_drawdown",
    "win_rate",
    "profit_factor",
    "total_trades",
    "total_volume_usdt",
    "annual_return",
    "daily_pnl_mean",
    "daily_pnl_std",
    "daily_pnl_min",
    "daily_pnl_max",
    "daily_pnl_median",
    "n_days",
}


class TestSignalBacktestEngine:
    def setup_method(self) -> None:
        self.engine = SignalBacktestEngine(
            init_cash=10_000.0, fees=0.0, slippage=0.0
        )

    def _make_simple_df(
        self, prices: list[float], signals: list[int]
    ) -> pd.DataFrame:
        n = len(prices)
        return pd.DataFrame(
            {
                "timestamp": pd.date_range(
                    "2024-01-01",
                    periods=n,
                    freq="h",
                    tz=UTC,
                ),
                "close": prices,
                "signal": signals,
            }
        )

    def test_returns_backtest_result(self) -> None:
        prices = [100.0] * 20
        signals = [0] * 20
        df = self._make_simple_df(prices, signals)
        result = self.engine.run(df)
        assert isinstance(result, BacktestResult)
        assert result.mode == "signal"
        assert len(result.equity_curve) == 20

    def test_all_long_on_uptrend(self) -> None:
        prices = [100.0 + i * 1.0 for i in range(50)]
        signals = [1] * 50
        df = self._make_simple_df(prices, signals)
        result = self.engine.run(df)
        assert result.metrics["total_return"] > 0

    def test_no_signal_no_trades(self) -> None:
        prices = [100.0 + i for i in range(50)]
        signals = [0] * 50
        df = self._make_simple_df(prices, signals)
        result = self.engine.run(df)
        assert result.metrics["total_trades"] == 0

    def test_fees_reduce_returns(self) -> None:
        prices = [100.0 + i * 1.0 for i in range(50)]
        signals = [1] * 50
        df = self._make_simple_df(prices, signals)
        result_no_fees = self.engine.run(df)

        engine_fees = SignalBacktestEngine(
            init_cash=10_000.0, fees=0.01, slippage=0.0
        )
        result_fees = engine_fees.run(df)
        assert (
            result_fees.metrics["total_return"]
            < result_no_fees.metrics["total_return"]
        )

    def test_metrics_keys(self) -> None:
        prices = [100.0] * 20
        signals = [0] * 20
        df = self._make_simple_df(prices, signals)
        result = self.engine.run(df)
        assert set(result.metrics.keys()) == EXPECTED_KEYS


class TestVolumeBacktestEngine:
    def test_returns_backtest_result(self) -> None:
        prices = [100.0] * 20
        signals = [1] * 20
        df = _make_ohlcv_df(prices, signals)
        engine = VolumeBacktestEngine(
            init_cash=10_000.0,
            fees=0.0,
            slippage=0.0,
            position_size_usdt=200.0,
        )
        result = engine.run(df)
        assert isinstance(result, BacktestResult)
        assert result.mode == "volume"
        assert len(result.equity_curve) == 20
        assert result.timestamps is not None

    def test_reentry_after_sl(self) -> None:
        prices = [100.0] * 5 + [95.0] + [100.0] * 14
        signals = [1] * 20
        df = _make_ohlcv_df(prices, signals, spread=0.3)
        engine = VolumeBacktestEngine(
            init_cash=10_000.0,
            fees=0.0,
            slippage=0.0,
            position_size_usdt=200.0,
            stop_loss=0.01,
        )
        result = engine.run(df)
        assert result.metrics["total_trades"] >= 2

    def test_direction_change(self) -> None:
        prices = [100.0] * 20
        signals = [1] * 10 + [-1] * 10
        df = _make_ohlcv_df(prices, signals)
        engine = VolumeBacktestEngine(
            init_cash=10_000.0,
            fees=0.0,
            slippage=0.0,
            position_size_usdt=200.0,
        )
        result = engine.run(df)
        assert result.metrics["total_trades"] >= 2

    def test_always_in_position(self) -> None:
        n = 200
        rng = np.random.default_rng(42)
        prices = [100.0]
        for _ in range(n - 1):
            prices.append(prices[-1] + rng.normal(0, 0.5))
        signals = [1] * n
        df = _make_ohlcv_df(prices, signals, spread=1.0)
        engine = VolumeBacktestEngine(
            init_cash=10_000.0,
            fees=0.0,
            slippage=0.0,
            position_size_usdt=200.0,
            stop_loss=0.005,
            take_profit=0.004,
        )
        result = engine.run(df)
        assert result.metrics["total_trades"] > 10

    def test_metrics_keys(self) -> None:
        prices = [100.0] * 20
        signals = [1] * 20
        df = _make_ohlcv_df(prices, signals)
        engine = VolumeBacktestEngine(
            init_cash=10_000.0,
            fees=0.0,
            slippage=0.0,
            position_size_usdt=200.0,
        )
        result = engine.run(df)
        assert set(result.metrics.keys()) == EXPECTED_KEYS

    def test_more_trades_than_signal_mode(self) -> None:
        n = 100
        rng = np.random.default_rng(99)
        prices = [100.0]
        for _ in range(n - 1):
            prices.append(prices[-1] + rng.normal(0, 0.3))
        signals = [1] * n
        df = _make_ohlcv_df(prices, signals, spread=0.8)

        sig_engine = SignalBacktestEngine(
            init_cash=10_000.0,
            fees=0.0,
            slippage=0.0,
            position_size_usdt=200.0,
            stop_loss=0.005,
            take_profit=0.004,
        )
        vol_engine = VolumeBacktestEngine(
            init_cash=10_000.0,
            fees=0.0,
            slippage=0.0,
            position_size_usdt=200.0,
            stop_loss=0.005,
            take_profit=0.004,
        )
        sig_r = sig_engine.run(df)
        vol_r = vol_engine.run(df)
        assert (
            vol_r.metrics["total_trades"]
            >= sig_r.metrics["total_trades"]
        )

    def test_no_signal_no_trades(self) -> None:
        prices = [100.0] * 20
        signals = [0] * 20
        df = _make_ohlcv_df(prices, signals)
        engine = VolumeBacktestEngine(
            init_cash=10_000.0,
            fees=0.0,
            slippage=0.0,
            position_size_usdt=200.0,
        )
        result = engine.run(df)
        assert result.metrics["total_trades"] == 0
