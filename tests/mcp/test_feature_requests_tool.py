"""Feature request MCP tool tests using in-process FastMCP dispatch."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from mcp.server.fastmcp import FastMCP

from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.app.contracts.research import RecordFeatureRequestResponse
from tradingdev.app.feature_request_service import FeatureRequestService
from tradingdev.mcp.tools import feature_requests

if TYPE_CHECKING:
    from pathlib import Path


def test_record_feature_request_returns_success(tmp_path: Path) -> None:
    service = FeatureRequestService(
        workspace=WorkspacePaths(tmp_path / "workspace"),
    )
    mcp = FastMCP("feature-request-test")
    feature_requests.register(mcp, service)

    async def check() -> None:
        result = await mcp.call_tool(
            "record_feature_request",
            {
                "title": "Need live trading",
                "description": "Support live order execution.",
                "source_tool": "start_live_trading",
            },
        )
        assert isinstance(result, tuple)
        response = RecordFeatureRequestResponse.model_validate(result[1])
        assert response.success is True
        assert response.feature_request.success is True
        assert response.feature_request.request_id
        assert "unsupported" not in result[1]

    asyncio.run(check())
    metadata = service.list_requests()[0]["metadata"]
    assert isinstance(metadata, dict)
    assert metadata["source_tool"] == "start_live_trading"
