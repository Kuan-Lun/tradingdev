"""Run MCP tools."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

    from tradingdev.app.run_service import RunService


def register(mcp: FastMCP, service: RunService) -> None:
    """Register run tools."""

    @mcp.tool()
    def list_runs() -> list[dict[str, Any]]:
        """List completed runs."""
        return service.list_runs()

    @mcp.tool()
    def get_run(run_id: str) -> dict[str, Any]:
        """Return one completed run."""
        return service.get_run(run_id)

    @mcp.tool()
    def compare_runs(run_ids: list[str]) -> dict[str, Any]:
        """Compare metrics for selected runs."""
        return service.compare_runs(run_ids)
