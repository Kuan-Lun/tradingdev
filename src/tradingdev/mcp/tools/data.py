"""Data MCP tools."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from tradingdev.domain.backtest.schemas import BacktestConfig

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
    ) -> dict[str, Any]:
        """Ensure OHLCV data for the requested range is cached."""
        bt_cfg = BacktestConfig(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            mode="volume",
        )
        dataset = service.load({"data": {"source": "binance_api"}}, bt_cfg)
        return {
            "success": True,
            "rows": len(dataset.frame),
            "processed_path": str(dataset.processed_path),
            "dataset_id": dataset.dataset_id,
        }
