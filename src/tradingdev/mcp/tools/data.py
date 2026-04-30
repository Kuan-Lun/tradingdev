"""Data MCP tools."""

from __future__ import annotations

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
    def inspect_dataset() -> dict[str, Any]:
        """Inspect the workspace data cache."""
        return service.inspect_dataset()

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
