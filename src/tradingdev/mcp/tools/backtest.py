"""Backtest MCP tools."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from tradingdev.mcp.schemas import BacktestInput

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

    from tradingdev.app.job_service import JobService


def register(mcp: FastMCP, service: JobService) -> None:
    """Register backtest execution tools."""

    @mcp.tool()
    def start_backtest(
        strategy_id: str,
        symbol: str,
        timeframe: str,
        start_date: str,
        end_date: str,
    ) -> dict[str, Any]:
        """Launch a simple background backtest job."""
        payload = BacktestInput(
            strategy_id=strategy_id,
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
        )
        return service.start_backtest(
            strategy_id=payload.strategy_id,
            symbol=payload.symbol,
            timeframe=payload.timeframe,
            start_date=payload.start_date,
            end_date=payload.end_date,
        )

    @mcp.tool()
    def start_walk_forward(
        strategy_id: str,
        symbol: str,
        timeframe: str,
        start_date: str,
        end_date: str,
    ) -> dict[str, Any]:
        """Launch a background walk-forward validation job."""
        payload = BacktestInput(
            strategy_id=strategy_id,
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
        )
        return service.start_walk_forward(
            strategy_id=payload.strategy_id,
            symbol=payload.symbol,
            timeframe=payload.timeframe,
            start_date=payload.start_date,
            end_date=payload.end_date,
        )
