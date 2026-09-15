"""Integration fixtures backed by the reusable real MCP process harness."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from tests.integration.mcp_harness import temporary_mcp_workspace

if TYPE_CHECKING:
    from collections.abc import Iterator

    from tests.integration.mcp_harness import MCPWorkspace


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def mcp_workspace() -> Iterator[MCPWorkspace]:
    with temporary_mcp_workspace() as workspace:
        yield workspace
