"""Optimization MCP tools."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from tradingdev.mcp.schemas import OptimizationInput

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

    from tradingdev.app.job_service import JobService
    from tradingdev.app.optimization_service import OptimizationService


def register(
    mcp: FastMCP,
    optimization_service: OptimizationService,
    job_service: JobService,
) -> None:
    """Register optimization tools."""

    @mcp.tool()
    def start_optimization(
        strategy_id: str,
        symbol: str,
        timeframe: str,
        param_ranges: dict[str, list[Any]],
        optimization_metric: str,
        train_start: str,
        train_end: str,
        test_start: str,
        test_end: str,
    ) -> dict[str, Any]:
        """Launch a parameter optimization job."""
        payload = OptimizationInput(
            strategy_id=strategy_id,
            symbol=symbol,
            timeframe=timeframe,
            param_ranges=param_ranges,
            optimization_metric=optimization_metric,
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
        )
        return optimization_service.start_optimization(
            strategy_id=payload.strategy_id,
            symbol=payload.symbol,
            timeframe=payload.timeframe,
            param_ranges=payload.param_ranges,
            optimization_metric=payload.optimization_metric,
            train_start=payload.train_start,
            train_end=payload.train_end,
            test_start=payload.test_start,
            test_end=payload.test_end,
        )

    @mcp.tool()
    def confirm_optimization(job_id: str) -> dict[str, Any]:
        """Confirm an optimization job after reviewing its estimate."""
        return job_service.confirm_optimization(job_id)
