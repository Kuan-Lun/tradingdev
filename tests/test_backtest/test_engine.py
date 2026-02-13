"""Tests for the backtest engine."""

from datetime import UTC

import numpy as np
import pandas as pd

from btc_strategy.backtest.engine import BacktestEngine


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


class TestBacktestEngine:
    def setup_method(self) -> None:
        self.engine = BacktestEngine(
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

    def test_all_long_on_uptrend(self) -> None:
        """All-long signal on uptrend should produce positive return."""
        prices = [100.0 + i * 1.0 for i in range(50)]
        signals = [1] * 50
        df = self._make_simple_df(prices, signals)
        metrics = self.engine.run(df)
        assert metrics["total_return"] > 0

    def test_no_signal_no_trades(self) -> None:
        """All-zero signal should result in zero trades."""
        prices = [100.0 + i for i in range(50)]
        signals = [0] * 50
        df = self._make_simple_df(prices, signals)
        metrics = self.engine.run(df)
        assert metrics["total_trades"] == 0

    def test_fees_reduce_returns(self) -> None:
        """Fees should reduce returns compared to zero fees."""
        prices = [100.0 + i * 1.0 for i in range(50)]
        signals = [1] * 50

        df = self._make_simple_df(prices, signals)
        metrics_no_fees = self.engine.run(df)

        engine_with_fees = BacktestEngine(init_cash=10_000.0, fees=0.01, slippage=0.0)
        metrics_with_fees = engine_with_fees.run(df)

        assert metrics_with_fees["total_return"] < metrics_no_fees["total_return"]

    def test_metrics_keys(self) -> None:
        """Verify all expected metric keys are present."""
        prices = [100.0] * 20
        signals = [0] * 20
        df = self._make_simple_df(prices, signals)
        metrics = self.engine.run(df)
        expected_keys = {
            "total_return",
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
        assert expected_keys == set(metrics.keys())


class TestVolumeMode:
    """Tests for the volume-mode backtest engine."""

    def test_reentry_after_sl(self) -> None:
        """After SL exit, engine should re-enter on the same bar."""
        # Price drops sharply to trigger SL, then recovers.
        # With signal always=1, volume mode should close via SL
        # and immediately re-enter.
        prices = [100.0] * 5 + [95.0] + [100.0] * 14
        signals = [1] * 20
        df = _make_ohlcv_df(prices, signals, spread=0.3)
        # SL at 1% → triggers when low <= 99.0 (entry ~100)
        engine = BacktestEngine(
            init_cash=10_000.0,
            fees=0.0,
            slippage=0.0,
            position_size_usdt=200.0,
            stop_loss=0.01,
            mode="volume",
        )
        metrics = engine.run(df)
        # Should have at least 2 trades: initial entry + re-entry
        assert metrics["total_trades"] >= 2

    def test_direction_change(self) -> None:
        """Direction change should close and reverse position."""
        prices = [100.0] * 20
        signals = [1] * 10 + [-1] * 10
        df = _make_ohlcv_df(prices, signals)
        engine = BacktestEngine(
            init_cash=10_000.0,
            fees=0.0,
            slippage=0.0,
            position_size_usdt=200.0,
            mode="volume",
        )
        metrics = engine.run(df)
        # At least 2 trades: long closed + short closed at end
        assert metrics["total_trades"] >= 2

    def test_always_in_position(self) -> None:
        """With tight SL/TP and all-1 signal, many trades."""
        n = 200
        # Volatile prices to trigger frequent SL/TP
        rng = np.random.default_rng(42)
        base = 100.0
        prices = [base]
        for _ in range(n - 1):
            prices.append(
                prices[-1] + rng.normal(0, 0.5)
            )
        signals = [1] * n
        df = _make_ohlcv_df(prices, signals, spread=1.0)
        engine = BacktestEngine(
            init_cash=10_000.0,
            fees=0.0,
            slippage=0.0,
            position_size_usdt=200.0,
            stop_loss=0.005,
            take_profit=0.004,
            mode="volume",
        )
        metrics = engine.run(df)
        # With tight SL/TP, should have many trades
        assert metrics["total_trades"] > 10

    def test_metrics_keys(self) -> None:
        """Volume mode should return the same metric keys."""
        prices = [100.0] * 20
        signals = [1] * 20
        df = _make_ohlcv_df(prices, signals)
        engine = BacktestEngine(
            init_cash=10_000.0,
            fees=0.0,
            slippage=0.0,
            position_size_usdt=200.0,
            mode="volume",
        )
        metrics = engine.run(df)
        expected_keys = {
            "total_return",
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
        assert expected_keys == set(metrics.keys())

    def test_more_trades_than_signal_mode(self) -> None:
        """Same signals should yield more trades in volume mode."""
        n = 100
        rng = np.random.default_rng(99)
        prices = [100.0]
        for _ in range(n - 1):
            prices.append(prices[-1] + rng.normal(0, 0.3))
        signals = [1] * n
        df = _make_ohlcv_df(prices, signals, spread=0.8)

        signal_engine = BacktestEngine(
            init_cash=10_000.0,
            fees=0.0,
            slippage=0.0,
            position_size_usdt=200.0,
            stop_loss=0.005,
            take_profit=0.004,
            mode="signal",
        )
        volume_engine = BacktestEngine(
            init_cash=10_000.0,
            fees=0.0,
            slippage=0.0,
            position_size_usdt=200.0,
            stop_loss=0.005,
            take_profit=0.004,
            mode="volume",
        )
        sig_metrics = signal_engine.run(df)
        vol_metrics = volume_engine.run(df)
        assert (
            vol_metrics["total_trades"]
            >= sig_metrics["total_trades"]
        )

    def test_no_signal_no_trades(self) -> None:
        """All-zero signals should produce zero trades."""
        prices = [100.0] * 20
        signals = [0] * 20
        df = _make_ohlcv_df(prices, signals)
        engine = BacktestEngine(
            init_cash=10_000.0,
            fees=0.0,
            slippage=0.0,
            position_size_usdt=200.0,
            mode="volume",
        )
        metrics = engine.run(df)
        assert metrics["total_trades"] == 0
