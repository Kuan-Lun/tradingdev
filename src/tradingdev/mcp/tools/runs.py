"""Run MCP tools."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mcp.types import ToolAnnotations

from tradingdev.app.contracts.common import ErrorResponse
from tradingdev.app.contracts.research import (
    CompareRunsResponse,
    RunRecord,
    RunResponse,
)

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

    from tradingdev.app.run_service import RunService


def register(mcp: FastMCP, service: RunService) -> None:
    """Register run tools."""

    @mcp.tool(
        annotations=ToolAnnotations(
            readOnlyHint=True,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=False,
        )
    )
    def list_runs() -> list[RunRecord]:
        """List completed runs; use a returned run_id with get_run or list_artifacts."""
        return [RunRecord.model_validate(row) for row in service.list_runs()]

    @mcp.tool(
        annotations=ToolAnnotations(
            readOnlyHint=True,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=False,
        )
    )
    def get_run(run_id: str) -> RunResponse | ErrorResponse:
        """Read a completed run by ID; use list_artifacts for its result files."""
        payload = service.get_run(run_id)
        if payload.get("success") is False:
            return ErrorResponse.model_validate(payload)
        return RunResponse.model_validate(payload)

    @mcp.tool(
        annotations=ToolAnnotations(
            readOnlyHint=True,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=False,
        )
    )
    def compare_runs(run_ids: list[str]) -> CompareRunsResponse | ErrorResponse:
        """Compare metrics for at least two completed IDs obtained from list_runs."""
        payload = service.compare_runs(run_ids)
        if payload.get("success") is False:
            return ErrorResponse.model_validate(payload)
        return CompareRunsResponse.model_validate(payload)
