"""MCP tool input/output DTOs."""

from __future__ import annotations

from pydantic import BaseModel


class SaveStrategyInput(BaseModel):
    """Input contract for saving a generated strategy."""

    strategy_id: str
    code: str
    yaml_config: str
    request_summary: str = ""


class ToolResult(BaseModel):
    """Generic MCP-friendly result."""

    success: bool
    message: str = ""
    error: str | None = None
