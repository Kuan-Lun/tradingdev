"""Reject misspelled MCP arguments before FastMCP's permissive parsing."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError

if TYPE_CHECKING:
    from collections.abc import Sequence

    from mcp.types import ContentBlock, Tool


class StrictFastMCP(FastMCP):
    """Keep the advertised argument names and dispatch boundary in agreement.

    FastMCP's generated argument models ignore extra fields, and its protocol
    handler disables JSON Schema input validation. These public method overrides
    reject unknown top-level arguments while preserving SDK type conversion,
    optional defaults, and the declared contents of dynamic parameter maps.
    """

    async def list_tools(self) -> list[Tool]:
        """Advertise closed argument objects without mutating SDK tool metadata."""
        return [
            tool.model_copy(
                update={
                    "inputSchema": tool.inputSchema | {"additionalProperties": False}
                }
            )
            for tool in await super().list_tools()
        ]

    async def call_tool(
        self, name: str, arguments: dict[str, Any]
    ) -> Sequence[ContentBlock] | dict[str, Any]:
        """Reject unknown keys before business logic or worker creation runs."""
        for tool in await self.list_tools():
            if tool.name == name:
                properties = tool.inputSchema.get("properties", {})
                unknown = sorted(arguments.keys() - properties.keys())
                if unknown:
                    allowed = ", ".join(sorted(properties)) or "(none)"
                    raise ToolError(
                        f"Unknown arguments for {name}: {', '.join(unknown)}. "
                        f"Allowed arguments: {allowed}"
                    )
                break
        return await super().call_tool(name, arguments)
