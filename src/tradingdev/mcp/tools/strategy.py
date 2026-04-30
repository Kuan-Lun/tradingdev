"""Strategy lifecycle MCP tools."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

    from mcp.server.fastmcp import FastMCP

    from tradingdev.app.strategy_service import StrategyService


def register(mcp: FastMCP, service: StrategyService, package_root: Path) -> None:
    """Register strategy lifecycle tools."""

    @mcp.tool()
    def get_strategy_contract() -> dict[str, str]:
        """Return reference code and YAML contract for generated strategies."""
        base_path = package_root / "domain" / "strategies" / "base.py"
        base_source = (
            base_path.read_text(encoding="utf-8") if base_path.exists() else ""
        )
        example_code = '''\
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
        example_yaml = """\
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
        return {
            "base_strategy_source": base_source,
            "example_strategy_code": example_code,
            "example_yaml_config": example_yaml,
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
                "smoke signal contract with structured diagnostics; "
                "dry_run_strategy returns signal_analysis and promotes validated "
                "code to runnable; "
                "start_backtest/start_walk_forward accept only runnable or promoted "
                "generated strategies and promoted bundled strategies."
            ),
        }

    @mcp.tool()
    def list_strategies() -> list[dict[str, Any]]:
        """List bundled and generated strategies."""
        return service.list_strategies()

    @mcp.tool()
    def get_strategy(strategy_id: str) -> dict[str, Any]:
        """Retrieve source, YAML config, and metadata for a strategy."""
        return service.get_strategy(strategy_id)

    @mcp.tool()
    def save_strategy(name: str, code: str, yaml_config: str) -> dict[str, Any]:
        """Save generated strategy code and YAML as a draft."""
        saved = service.save_draft(name, code, yaml_config)
        return {
            "success": saved.success,
            "strategy_id": saved.strategy_id,
            "py_path": saved.source_path,
            "yaml_path": saved.config_path,
            "status": saved.status,
            "error": saved.error,
        }

    @mcp.tool()
    def validate_strategy(strategy_id: str) -> dict[str, Any]:
        """Validate a generated strategy draft."""
        return service.validate(strategy_id)

    @mcp.tool()
    def dry_run_strategy(strategy_id: str) -> dict[str, Any]:
        """Run a signal-generation dry run for a validated strategy."""
        return service.dry_run(strategy_id)

    @mcp.tool()
    def promote_strategy(strategy_id: str) -> dict[str, Any]:
        """Promote a runnable generated strategy."""
        return service.promote(strategy_id)
