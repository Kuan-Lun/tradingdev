"""Optimization MCP tools."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mcp.types import ToolAnnotations
from pydantic import JsonValue, TypeAdapter

from tradingdev.app.contracts.jobs import (
    JobActionFailed,
    OptimizationConfirmed,
    OptimizationRejected,
    OptimizationStarted,
)
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

    @mcp.tool(
        annotations=ToolAnnotations(
            readOnlyHint=False,
            destructiveHint=True,
            idempotentHint=False,
            openWorldHint=True,
        )
    )
    def start_optimization(
        strategy_id: str,
        symbol: str,
        timeframe: str,
        param_ranges: dict[str, list[JsonValue]],
        optimization_metric: str,
        train_start: str,
        train_end: str,
        test_start: str,
        test_end: str,
        revision_id: str | None = None,
    ) -> OptimizationStarted | OptimizationRejected:
        """Launch optimization and wait for confirmation after estimating its cost.

        Pass the runnable revision_id; omission selects current at submission.
        The returned manifest_hash pins settings, grid, metric, and date splits.
        The strategy config must not contain walk-forward validation settings.
        Dates are inclusive UTC calendar days and must satisfy
        train_start < train_end < test_start < test_end, without overlap.
        Grid values override matching YAML parameters; other parameters stay fixed.
        Review get_job_status before calling confirm_optimization.
        May download data and replace partial caches.
        """
        payload = OptimizationInput(
            strategy_id=strategy_id,
            revision_id=revision_id,
            symbol=symbol,
            timeframe=timeframe,
            param_ranges=param_ranges,
            optimization_metric=optimization_metric,
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
        )
        result = optimization_service.start_optimization(
            strategy_id=payload.strategy_id,
            revision_id=payload.revision_id,
            symbol=payload.symbol,
            timeframe=payload.timeframe,
            param_ranges=payload.param_ranges,
            optimization_metric=payload.optimization_metric,
            train_start=payload.train_start,
            train_end=payload.train_end,
            test_start=payload.test_start,
            test_end=payload.test_end,
        )
        return TypeAdapter(OptimizationStarted | OptimizationRejected).validate_python(
            result
        )

    @mcp.tool(
        annotations=ToolAnnotations(
            readOnlyHint=False,
            destructiveHint=True,
            idempotentHint=True,
            openWorldHint=True,
        )
    )
    def confirm_optimization(job_id: str) -> OptimizationConfirmed | JobActionFailed:
        """Confirm an existing pending_confirmation job after user accepts its estimate.

        Poll get_job_status for progress; this does not create another job.
        Missing or altered execution manifests return execution_manifest_invalid;
        submit a new job instead of resuming legacy jobs without a manifest.
        """
        return TypeAdapter(OptimizationConfirmed | JobActionFailed).validate_python(
            job_service.confirm_optimization(job_id)
        )
