"""Feature request MCP tools."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

    from tradingdev.app.feature_request_service import FeatureRequestService


def register(
    mcp: FastMCP,
    service: FeatureRequestService,
) -> None:
    """Register feature-request tools."""

    @mcp.tool()
    def record_feature_request(
        title: str,
        description: str,
        source_tool: str = "",
    ) -> dict[str, object]:
        """Record an unsupported feature request artifact."""
        request = service.record(
            title=title,
            description=description,
            source_tool=source_tool or "record_feature_request",
        )
        return {
            "success": True,
            "message": "Feature request recorded.",
            "feature_request": request,
        }

    @mcp.tool()
    def list_feature_requests() -> list[dict[str, object]]:
        """List recorded feature request artifacts."""
        return service.list_requests()
