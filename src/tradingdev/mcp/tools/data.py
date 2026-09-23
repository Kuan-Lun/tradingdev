"""Data MCP tools."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from mcp.types import ToolAnnotations

from tradingdev.app.contracts.data import (
    CachedDataset,
    DatasetInspection,
    EnsureDataResponse,
)
from tradingdev.domain.data.crawlers.registry import available_sources

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

    from tradingdev.app.data_service import DataService


def register(mcp: FastMCP, service: DataService) -> None:
    """Register data tools."""

    @mcp.tool(
        annotations=ToolAnnotations(
            readOnlyHint=True,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=False,
        )
    )
    def list_available_data() -> list[CachedDataset]:
        """List cached OHLCV datasets; use inspect_dataset to inspect file contents."""
        return [
            CachedDataset.model_validate(row) for row in service.list_available_data()
        ]

    @mcp.tool(
        annotations=ToolAnnotations(
            readOnlyHint=True,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=False,
        )
    )
    def list_data_sources() -> list[str]:
        """List registered source names before selecting a source for ensure_data."""
        return available_sources()

    @mcp.tool(
        annotations=ToolAnnotations(
            readOnlyHint=True,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=False,
        )
    )
    def inspect_dataset(config_path: str | None = None) -> DatasetInspection:
        """Inspect local caches, optionally against an existing run config.

        This does not fetch data. Use ensure_data to acquire missing market data.
        """
        path = Path(config_path) if config_path else None
        return DatasetInspection.model_validate(service.inspect_dataset(path))

    @mcp.tool(
        annotations=ToolAnnotations(
            readOnlyHint=False,
            destructiveHint=True,
            idempotentHint=False,
            openWorldHint=True,
        )
    )
    def ensure_data(
        symbol: str,
        timeframe: str,
        start_date: str,
        end_date: str,
        source: str = "binance_vision",
    ) -> EnsureDataResponse:
        """Fetch and cache OHLCV data for the requested range when needed.

        Select a source from list_data_sources. Completed-year downloads may
        replace partial caches. Inspect the cache before starting a backtest.
        """
        dataset = service.ensure(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            source=source,
        )
        return EnsureDataResponse(
            success=True,
            rows=len(dataset.frame),
            processed_path=str(dataset.processed_path),
            dataset_id=dataset.dataset_id,
        )
