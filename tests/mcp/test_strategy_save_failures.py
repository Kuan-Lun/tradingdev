"""Regression coverage for generated saves rejected by the MCP boundary."""

from __future__ import annotations

import asyncio
from pathlib import Path

from jsonschema import Draft202012Validator
from mcp.server.fastmcp import FastMCP
from mcp.shared.memory import create_connected_server_and_client_session

from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.app.strategy_service import StrategyService
from tradingdev.mcp.tools.strategy import register

_PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "src" / "tradingdev"


def test_reserved_strategy_id_returns_a_structured_rejection(tmp_path: Path) -> None:
    workspace = WorkspacePaths(tmp_path / "workspace")
    service = StrategyService(workspace)
    server = FastMCP("strategy-save-failure-test")
    register(server, service, _PACKAGE_ROOT)

    async def check_rejected_save() -> None:
        # The SDK transport exercises a real MCP session and its error boundary,
        # while avoiding a process for this purely local contract regression.
        async with create_connected_server_and_client_session(server) as client:
            listed = await client.call_tool("list_strategies", {})
            assert not listed.isError
            assert listed.structuredContent is not None
            bundled = next(
                item
                for item in listed.structuredContent["result"]
                if item["kind"] == "bundled"
            )
            strategy_id = bundled["strategy_id"]
            before = await client.call_tool(
                "get_strategy", {"strategy_id": strategy_id}
            )
            assert not before.isError
            assert before.structuredContent is not None
            source = before.structuredContent["result"]
            tools = await client.list_tools()
            schema = next(
                tool.outputSchema
                for tool in tools.tools
                if tool.name == "save_strategy"
            )
            assert schema is not None
            Draft202012Validator.check_schema(schema)

            result = await client.call_tool(
                "save_strategy",
                {
                    "strategy_id": strategy_id,
                    "code": source["source_code"],
                    "yaml_config": source["yaml_config"],
                },
            )

            assert not result.isError, result.model_dump(mode="json")
            assert result.structuredContent is not None
            Draft202012Validator(schema).validate(result.structuredContent)
            rejected = result.structuredContent["result"]
            assert rejected["success"] is False
            assert rejected["code"] == "reserved_strategy_id"
            assert rejected["strategy_id"] == strategy_id
            assert rejected["status"] == "rejected"
            assert rejected["revision_id"] is None
            assert rejected["py_path"] == rejected["yaml_path"] == ""
            assert rejected["error"]

            after = await client.call_tool("get_strategy", {"strategy_id": strategy_id})
            assert not after.isError
            assert after.structuredContent == before.structuredContent
            still_listed = await client.call_tool("list_strategies", {})
            assert not still_listed.isError
            assert still_listed.structuredContent == listed.structuredContent
            assert not any(workspace.generated_strategies.iterdir())

    asyncio.run(check_rejected_save())
