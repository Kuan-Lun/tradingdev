"""Data tool contracts through in-process MCP dispatch with a local cache."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import pandas as pd
import pytest
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError
from pydantic import TypeAdapter, ValidationError

from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.app.contracts.data import (
    CachedDataset,
    DatasetInspection,
    EnsureDataResponse,
)
from tradingdev.app.data_service import DataService
from tradingdev.mcp.tools import data

if TYPE_CHECKING:
    from pathlib import Path

    from pytest import MonkeyPatch


def test_data_tools_validate_local_cache_and_inspection(tmp_path: Path) -> None:
    workspace = WorkspacePaths(tmp_path / "workspace")
    service = DataService(workspace)
    path = workspace.processed_data / "btcusdt_1h_2024.parquet"
    pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC"),
            "open": [100.0, 101.0, 102.0],
            "high": [101.0, 102.0, 103.0],
            "low": [99.0, 100.0, 101.0],
            "close": [100.5, 101.5, 102.5],
            "volume": [10.0, 11.0, 12.0],
        }
    ).to_parquet(path, index=False)
    config_path = tmp_path / "fixture.yaml"
    config_path.write_text(
        "backtest:\n  symbol: BTC/USDT\n  timeframe: 1h\n"
        '  start_date: "2024-01-01"\n  end_date: "2024-01-31"\n'
        "  init_cash: 10000\n"
        "data:\n  requirements:\n    market:\n      symbol: BTC/USDT\n"
        "      timeframe: 1h\n    features:\n      - type: custom\n"
        "        source: local\n        column: close\n"
        f'        path: "{path}"\n',
        encoding="utf-8",
    )
    mcp = FastMCP("data-contract-test")
    data.register(mcp, service)

    async def call(name: str, arguments: dict[str, object]) -> dict[str, object]:
        result = await mcp.call_tool(name, arguments)
        assert isinstance(result, tuple)
        payload = result[1]
        assert isinstance(payload, dict)
        return payload

    async def check() -> None:
        listed = TypeAdapter(list[CachedDataset]).validate_python(
            (await call("list_available_data", {}))["result"]
        )
        assert listed[0].years_available == [2024]
        basic = DatasetInspection.model_validate(await call("inspect_dataset", {}))
        assert basic.requirements is None
        detailed = DatasetInspection.model_validate(
            await call("inspect_dataset", {"config_path": str(config_path)})
        )
        assert detailed.market is not None
        assert detailed.market.rows == 3
        assert detailed.requirements is not None
        assert detailed.requirements.features[0].raw_path is None
        assert detailed.features[0].missing_values == 0
        available = EnsureDataResponse.model_validate(
            await call(
                "ensure_data",
                {
                    "symbol": "BTC/USDT",
                    "timeframe": "1h",
                    "start_date": "2024-01-01",
                    "end_date": "2024-01-31",
                },
            )
        )
        assert available.rows == 3
        assert available.processed_path == str(path)

    asyncio.run(check())


def test_data_tool_rejects_undeclared_cache_fields(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    service = DataService(WorkspacePaths(tmp_path / "workspace"))
    monkeypatch.setattr(
        service,
        "list_available_data",
        lambda: [{"symbol": "BTC", "timeframe": "1h", "years_available": [], "bad": 1}],
    )
    mcp = FastMCP("data-drift-test")
    data.register(mcp, service)
    with pytest.raises(ToolError, match="extra_forbidden"):
        asyncio.run(mcp.call_tool("list_available_data", {}))


def test_data_contract_rejects_missing_required_fields() -> None:
    with pytest.raises(ValidationError, match="dataset_id"):
        EnsureDataResponse.model_validate({"rows": 0, "processed_path": "/unused"})
