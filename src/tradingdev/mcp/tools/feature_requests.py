"""Feature request MCP tools."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

    from tradingdev.app.capability_service import CapabilityService
    from tradingdev.app.feature_request_service import FeatureRequestService


def register(
    mcp: FastMCP,
    service: FeatureRequestService,
    capability_service: CapabilityService,
) -> None:
    """Register feature-request tools."""

    @mcp.tool()
    def record_feature_request(
        title: str,
        description: str,
        source_tool: str = "",
    ) -> dict[str, object]:
        """Record an unsupported feature request artifact."""
        return capability_service.unsupported(
            tool=source_tool or "record_feature_request",
            reason="Requested capability is not currently supported.",
            feature_title=title,
            feature_description=description,
        )

    @mcp.tool()
    def list_feature_requests() -> list[dict[str, object]]:
        """List recorded feature request artifacts."""
        return service.list_requests()
