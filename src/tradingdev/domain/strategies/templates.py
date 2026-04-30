"""Strategy contract templates for generated strategies."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


def strategy_contract_payload(package_root: Path) -> dict[str, str]:
    """Return source and YAML templates for generated strategies."""
    base_path = package_root / "domain" / "strategies" / "base.py"
    base_source = base_path.read_text(encoding="utf-8") if base_path.exists() else ""
    return {
        "base_strategy_source": base_source,
        "example_strategy_code": _EXAMPLE_CODE,
        "example_yaml_config": _EXAMPLE_YAML,
        "api_reference": (
            "Generated strategies must inherit BaseStrategy, return a DataFrame "
            "with signal values limited to -1, 0, and 1, and avoid mutating the "
            "input DataFrame. Use tradingdev.domain.indicators for built-in "
            "indicators and tradingdev.shared.utils.logger for logging. Allowed "
            "imports are restricted to a small Python/pandas/numpy/tradingdev "
            "allowlist."
        ),
        "lifecycle": (
            "save_strategy stores a draft; validate_strategy runs static checks, "
            "restricted import checks, ruff, mypy, inheritance checks, and a "
            "smoke signal contract with structured diagnostics; validate_strategy "
            "and dry_run_strategy currently execute generated Python code. "
            "Sandboxed execution isolation is required future work. "
            "dry_run_strategy returns signal_analysis and promotes validated "
            "code to runnable; start_backtest/start_walk_forward accept only "
            "runnable or promoted generated strategies and promoted bundled "
            "strategies."
        ),
    }


_EXAMPLE_CODE = '''\
"""Example: Simple moving-average crossover strategy."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd

from tradingdev.domain.strategies.base import BaseStrategy

if TYPE_CHECKING:
    from tradingdev.domain.backtest.base_engine import BaseBacktestEngine


class SmaCrossoverStrategy(BaseStrategy):
    """Buy when fast SMA crosses above slow SMA, sell on reverse."""

    def __init__(
        self,
        backtest_engine: BaseBacktestEngine | None = None,
        fast_period: int = 10,
        slow_period: int = 30,
    ) -> None:
        self._engine = backtest_engine
        self._fast_period = fast_period
        self._slow_period = slow_period

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        fast = result["close"].rolling(self._fast_period).mean()
        slow = result["close"].rolling(self._slow_period).mean()
        fast_prev = fast.shift(1)
        slow_prev = slow.shift(1)

        result["signal"] = 0
        result.loc[(fast > slow) & (fast_prev <= slow_prev), "signal"] = 1
        result.loc[(fast < slow) & (fast_prev >= slow_prev), "signal"] = -1
        return result

    def get_parameters(self) -> dict[str, Any]:
        return {
            "fast_period": self._fast_period,
            "slow_period": self._slow_period,
        }
'''

_EXAMPLE_YAML = """\
strategy:
  id: "sma_crossover"
  version: "0.1.0"
  class_name: "SmaCrossoverStrategy"
  source_path: "workspace/generated_strategies/sma_crossover.py"
  description: "Simple moving-average crossover"
  parameters:
    fast_period: 10
    slow_period: 30

backtest:
  symbol: "BTC/USDT"
  timeframe: "1h"
  start_date: "2024-01-01"
  end_date: "2024-12-31"
  init_cash: 10000.0
  fees: 0.0006
  slippage: 0.0005
  mode: "signal"

data:
  source: "binance_api"
  requirements:
    market:
      source: "binance_api"
      symbol: "BTC/USDT"
      timeframe: "1h"
    features: []
"""
