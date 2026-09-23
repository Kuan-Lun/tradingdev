"""Artifact MCP tools."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mcp.types import ToolAnnotations

from tradingdev.app.contracts.common import ErrorResponse
from tradingdev.app.contracts.research import ArtifactRecord, ArtifactResponse

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

    from tradingdev.app.artifact_service import ArtifactService


def register(mcp: FastMCP, service: ArtifactService) -> None:
    """Register artifact tools."""

    @mcp.tool(
        annotations=ToolAnnotations(
            readOnlyHint=True,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=False,
        )
    )
    def list_artifacts(run_id: str | None = None) -> list[ArtifactRecord]:
        """List local artifact metadata; use get_artifact to retrieve text content."""
        return [
            ArtifactRecord.model_validate(row) for row in service.list_artifacts(run_id)
        ]

    @mcp.tool(
        annotations=ToolAnnotations(
            readOnlyHint=True,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=False,
        )
    )
    def get_artifact(
        artifact_id: str,
        include_content: bool = False,
    ) -> ArtifactResponse | ErrorResponse:
        """Read an ID from list_artifacts, optionally including UTF-8 file content.

        Binary artifacts support metadata lookup only.
        """
        payload = service.get_artifact(artifact_id, include_content=include_content)
        if payload.get("success") is False:
            return ErrorResponse.model_validate(payload)
        return ArtifactResponse.model_validate(payload)
