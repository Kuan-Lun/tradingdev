"""Backtest schema tests."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from tradingdev.domain.backtest.schemas import BacktestConfig, BacktestRunConfig


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


def test_backtest_config_rejects_end_before_start() -> None:
    with pytest.raises(ValidationError, match="end_date must be after start_date"):
        BacktestConfig(
            symbol="BTC/USDT",
            timeframe="1h",
            start_date="2024-02-01",
            end_date="2024-01-01",
            init_cash=10_000.0,
        )


def test_backtest_config_rejects_equal_start_and_end() -> None:
    with pytest.raises(ValidationError, match="end_date must be after start_date"):
        BacktestConfig(
            symbol="BTC/USDT",
            timeframe="1h",
            start_date="2024-01-01",
            end_date="2024-01-01",
            init_cash=10_000.0,
        )
