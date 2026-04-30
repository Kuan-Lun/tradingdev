"""Backtest MCP tools."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

    from tradingdev.app.job_service import JobService


def register(mcp: FastMCP, service: JobService) -> None:
    """Register backtest execution tools."""

    @mcp.tool()
    def start_backtest(
        strategy_name: str,
        symbol: str,
        timeframe: str,
        start_date: str,
        end_date: str,
    ) -> dict[str, Any]:
        """Launch a simple background backtest job."""
        return service.start_backtest(
            strategy_id=strategy_name,
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
        )

    @mcp.tool()
    def start_walk_forward(
        strategy_name: str,
        symbol: str,
        timeframe: str,
        start_date: str,
        end_date: str,
    ) -> dict[str, Any]:
        """Launch a background walk-forward validation job."""
        return service.start_walk_forward(
            strategy_id=strategy_name,
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
        )
