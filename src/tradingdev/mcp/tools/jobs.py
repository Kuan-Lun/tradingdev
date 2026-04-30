"""Job MCP tools."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

    from tradingdev.app.job_service import JobService


def register(mcp: FastMCP, service: JobService) -> None:
    """Register job tools."""

    @mcp.tool()
    def get_job_status(job_id: str) -> dict[str, Any]:
        """Return the current status of a background job."""
        return service.get_job_status(job_id)

    @mcp.tool()
    def list_jobs() -> list[dict[str, Any]]:
        """List background jobs."""
        return service.list_jobs()

    @mcp.tool()
    def cancel_job(job_id: str) -> dict[str, Any]:
        """Cancel a queued or running background job."""
        return service.cancel_job(job_id)
