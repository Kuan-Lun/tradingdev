"""Tests for the backtest engine."""

from datetime import UTC

import pandas as pd

from btc_strategy.backtest.engine import BacktestEngine


class TestBacktestEngine:
    def setup_method(self) -> None:
        self.engine = BacktestEngine(init_cash=10_000.0, fees=0.0, slippage=0.0)

    def _make_simple_df(self, prices: list[float], signals: list[int]) -> pd.DataFrame:
        n = len(prices)
        return pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=n, freq="h", tz=UTC),
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
            "annual_return",
        }
        assert expected_keys == set(metrics.keys())
