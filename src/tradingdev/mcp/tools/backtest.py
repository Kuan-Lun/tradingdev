"""Backtest MCP tools."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mcp.types import ToolAnnotations
from pydantic import TypeAdapter

from tradingdev.app.contracts.jobs import BacktestRejected, BacktestStarted
from tradingdev.mcp.schemas import BacktestInput

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

    from tradingdev.app.job_service import JobService


def register(mcp: FastMCP, service: JobService) -> None:
    """Register backtest execution tools."""

    @mcp.tool(
        annotations=ToolAnnotations(
            readOnlyHint=False,
            destructiveHint=True,
            idempotentHint=False,
            openWorldHint=True,
        )
    )
    def start_backtest(
        strategy_id: str,
        symbol: str,
        timeframe: str,
        start_date: str,
        end_date: str,
    ) -> BacktestStarted | BacktestRejected:
        """Launch a backtest for a runnable/promoted strategy without validation folds.

        An empty job_id and code mean no job was created. Otherwise poll
        get_job_status, then use get_run after completion. May download data
        and replace partial caches.
        """
        payload = BacktestInput(
            strategy_id=strategy_id,
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
        )
        result = service.start_backtest(
            strategy_id=payload.strategy_id,
            symbol=payload.symbol,
            timeframe=payload.timeframe,
            start_date=payload.start_date,
            end_date=payload.end_date,
        )
        return TypeAdapter(BacktestStarted | BacktestRejected).validate_python(result)

    @mcp.tool(
        annotations=ToolAnnotations(
            readOnlyHint=False,
            destructiveHint=True,
            idempotentHint=False,
            openWorldHint=True,
        )
    )
    def start_walk_forward(
        strategy_id: str,
        symbol: str,
        timeframe: str,
        start_date: str,
        end_date: str,
    ) -> BacktestStarted | BacktestRejected:
        """Launch walk-forward for a runnable/promoted strategy with validation config.

        An empty job_id and code mean no job was created. Otherwise poll
        get_job_status, then use get_run after completion. May download data
        and replace partial caches.
        """
        payload = BacktestInput(
            strategy_id=strategy_id,
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
        )
        result = service.start_walk_forward(
            strategy_id=payload.strategy_id,
            symbol=payload.symbol,
            timeframe=payload.timeframe,
            start_date=payload.start_date,
            end_date=payload.end_date,
        )
        return TypeAdapter(BacktestStarted | BacktestRejected).validate_python(result)
