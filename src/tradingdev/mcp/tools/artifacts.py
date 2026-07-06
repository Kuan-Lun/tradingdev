"""Artifact MCP tools."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

    from tradingdev.app.artifact_service import ArtifactService


def register(mcp: FastMCP, service: ArtifactService) -> None:
    """Register artifact tools."""

    @mcp.tool()
    def list_artifacts(run_id: str | None = None) -> list[dict[str, Any]]:
        """List artifact metadata."""
        return service.list_artifacts(run_id)

    @mcp.tool()
    def get_artifact(
        artifact_id: str,
        include_content: bool = False,
    ) -> dict[str, Any]:
        """Return artifact metadata and optional text content."""
        return service.get_artifact(artifact_id, include_content=include_content)
