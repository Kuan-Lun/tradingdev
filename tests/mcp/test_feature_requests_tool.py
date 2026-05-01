"""Feature request MCP tool tests."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.app.feature_request_service import FeatureRequestService
from tradingdev.mcp.tools import feature_requests

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


class _FakeMCP:
    def __init__(self) -> None:
        self.tools: dict[str, Callable[..., object]] = {}

    def tool(self) -> Callable[[Callable[..., object]], Callable[..., object]]:
        def decorator(fn: Callable[..., object]) -> Callable[..., object]:
            self.tools[fn.__name__] = fn
            return fn

        return decorator


def test_record_feature_request_returns_success(tmp_path: Path) -> None:
    service = FeatureRequestService(
        workspace=WorkspacePaths(tmp_path / "workspace"),
    )
    mcp = _FakeMCP()

    feature_requests.register(
        cast("Any", mcp),
        service,
    )

    result = cast(
        "dict[str, object]",
        mcp.tools["record_feature_request"](
            "Need live trading",
            "Support live order execution.",
            "start_live_trading",
        ),
    )

    assert result["success"] is True
    assert "unsupported" not in result
    feature_request = cast("dict[str, object]", result["feature_request"])
    assert feature_request["success"] is True
    assert feature_request["request_id"]
    metadata = cast("dict[str, object]", service.list_requests()[0]["metadata"])
    assert metadata["source_tool"] == "start_live_trading"
