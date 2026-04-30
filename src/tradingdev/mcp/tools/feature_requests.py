"""Feature request MCP tools."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

    from tradingdev.app.feature_request_service import FeatureRequestService


def register(mcp: FastMCP, service: FeatureRequestService) -> None:
    """Register feature-request tools."""

    @mcp.tool()
    def record_feature_request(
        title: str,
        description: str,
        source_tool: str = "",
    ) -> dict[str, str | bool]:
        """Record an unsupported feature request artifact."""
        return service.record(
            title=title,
            description=description,
            source_tool=source_tool,
        )

    @mcp.tool()
    def list_feature_requests() -> list[dict[str, object]]:
        """List recorded feature request artifacts."""
        return service.list_requests()
