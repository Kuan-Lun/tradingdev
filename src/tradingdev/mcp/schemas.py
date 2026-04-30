"""MCP tool input/output DTOs."""

from __future__ import annotations

from pydantic import BaseModel


class SaveStrategyInput(BaseModel):
    """Input contract for saving a generated strategy."""

    strategy_id: str
    code: str
    yaml_config: str
    request_summary: str = ""


class ToolResult(BaseModel):
    """Generic MCP-friendly result."""

    success: bool
    message: str = ""
    error: str | None = None


class BacktestInput(BaseModel):
    """Input contract for starting a backtest-like job."""

    strategy_id: str
    symbol: str
    timeframe: str
    start_date: str
    end_date: str


class OptimizationInput(BaseModel):
    """Input contract for starting parameter optimization."""

    strategy_id: str
    symbol: str
    timeframe: str
    param_ranges: dict[str, list[object]]
    optimization_metric: str
    train_start: str
    train_end: str
    test_start: str
    test_end: str
