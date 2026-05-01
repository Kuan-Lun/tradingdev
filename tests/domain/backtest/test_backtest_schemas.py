"""Backtest schema tests."""

from __future__ import annotations

from tradingdev.domain.backtest.schemas import BacktestRunConfig


def test_backtest_run_config_accepts_random_seed() -> None:
    config = BacktestRunConfig.model_validate(
        {
            "random_seed": 42,
            "strategy": {"id": "fixture", "parameters": {}},
            "backtest": {
                "symbol": "BTC/USDT",
                "timeframe": "1h",
                "start_date": "2024-01-01",
                "end_date": "2024-01-02",
                "init_cash": 10000.0,
                "random_seed": 7,
            },
        }
    )

    assert config.random_seed == 42
    assert config.backtest.random_seed == 7
