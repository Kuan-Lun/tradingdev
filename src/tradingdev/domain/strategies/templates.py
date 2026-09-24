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
            "input DataFrame. Express configurable settings as constructor "
            "keyword arguments with finite JSON values, including any defaults. "
            "Execution captures these values at submission. backtest_engine and "
            "parallel_config are reserved application injections and cannot be "
            "overridden in strategy.parameters. Use tradingdev.domain.indicators "
            "(sma, ema, rsi, "
            "macd, bollinger_bands, atr, adx, stochastic) for standard "
            "indicators and tradingdev.shared.utils.logger for logging. "
            "The indicator facade uses TA-Lib and preserves its warm-up and "
            "NaN behavior; keep signals flat until all required values are "
            "finite, and never backfill indicators from future rows. "
            "talib may be imported directly: pass float64 NumPy arrays to its "
            "Function API, unpack multi-output tuples (MACD: line, signal, "
            "histogram; BBANDS: upper, middle, lower; STOCH: k, d), and align "
            "returned arrays with the input index. Allowed imports are "
            "restricted to a small Python/pandas/numpy/talib/tradingdev "
            "allowlist; pandas_ta is no longer supported. Ruff treats "
            "tradingdev as first-party: separate its imports from third-party "
            "numpy, pandas, and talib imports with a blank line."
        ),
        "lifecycle": (
            "save_strategy creates an immutable source/config revision and returns "
            "revision_id. Pass that revision_id to validate_strategy, "
            "dry_run_strategy, promote_strategy, get_strategy, and execution tools. "
            "Saving a repair creates a new draft revision; earlier revisions and "
            "their validation evidence remain available. Omitting revision_id "
            "selects current once per operation. validate_strategy runs static checks, "
            "restricted import checks, ruff, mypy, inheritance checks, and the "
            "shared signal-contract gate on a short fixture with structured "
            "diagnostics; dry_run_strategy accepts only validated strategies and "
            "re-runs the same signal-contract gate on a longer fixture, returns "
            "signal_analysis, and marks the strategy runnable; promote_strategy "
            "marks a runnable strategy promoted. validate_strategy and "
            "dry_run_strategy currently execute generated Python code; sandboxed "
            "execution isolation is required future work. Execution accepts only "
            "runnable or promoted strategies and enforces this gate at execution "
            "time on every entry point (MCP jobs, CLI, workers)."
        ),
    }


_EXAMPLE_CODE = '''\
"""Example: Simple moving-average crossover strategy."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from tradingdev.domain import indicators
from tradingdev.domain.strategies.base import BaseStrategy

if TYPE_CHECKING:
    import pandas as pd

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
        fast = indicators.sma(result["close"], self._fast_period)
        slow = indicators.sma(result["close"], self._slow_period)
        fast_prev = fast.shift(1)
        slow_prev = slow.shift(1)
        ready = (
            np.isfinite(fast)
            & np.isfinite(slow)
            & np.isfinite(fast_prev)
            & np.isfinite(slow_prev)
        )

        result["signal"] = 0
        result.loc[ready & (fast > slow) & (fast_prev <= slow_prev), "signal"] = 1
        result.loc[ready & (fast < slow) & (fast_prev >= slow_prev), "signal"] = -1
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
  # market_type: "futures/um"  # binance_vision only; use "spot" for spot
  requirements:
    market:
      # source selects the data crawler: "binance_vision" (default),
      # "binance_api" (ccxt), or "yahoo_finance" (stocks/futures/indices/FX,
      # Yahoo symbols such as "AAPL", "ES=F", "^GSPC")
      source: "binance_vision"
      symbol: "BTC/USDT"
      timeframe: "1h"
    features: []
"""
