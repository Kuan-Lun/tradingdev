"""Data MCP tools."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from tradingdev.domain.data.crawlers.registry import available_sources

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

    from tradingdev.app.data_service import DataService


def register(mcp: FastMCP, service: DataService) -> None:
    """Register data tools."""

    @mcp.tool()
    def list_available_data() -> list[dict[str, Any]]:
        """List cached OHLCV datasets."""
        return service.list_available_data()

    @mcp.tool()
    def list_data_sources() -> list[str]:
        """List registered market data source names."""
        return available_sources()

    @mcp.tool()
    def inspect_dataset(config_path: str | None = None) -> dict[str, Any]:
        """Inspect the workspace data cache."""
        path = Path(config_path) if config_path else None
        return service.inspect_dataset(path)

    @mcp.tool()
    def ensure_data(
        symbol: str,
        timeframe: str,
        start_date: str,
        end_date: str,
        source: str = "binance_vision",
    ) -> dict[str, Any]:
        """Ensure OHLCV data for the requested range is cached.

        ``source`` selects the market data crawler (see list_data_sources).
        """
        dataset = service.ensure(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            source=source,
        )
        return {
            "success": True,
            "rows": len(dataset.frame),
            "processed_path": str(dataset.processed_path),
            "dataset_id": dataset.dataset_id,
        }
