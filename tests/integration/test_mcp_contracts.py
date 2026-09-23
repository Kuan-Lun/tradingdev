"""Verify advertised contracts and error boundaries over real stdio MCP."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from jsonschema import Draft202012Validator, ValidationError, validate

if TYPE_CHECKING:
    from tests.integration.mcp_harness import MCPWorkspace

pytestmark = [pytest.mark.integration, pytest.mark.anyio]

_READ_ONLY_TOOLS = {
    "get_strategy_contract",
    "list_strategies",
    "get_strategy",
    "list_available_data",
    "list_data_sources",
    "inspect_dataset",
    "list_jobs",
    "list_runs",
    "get_run",
    "compare_runs",
    "list_artifacts",
    "get_artifact",
    "list_feature_requests",
}

# readOnly, destructive, idempotent, openWorld: expected public behavior,
# intentionally independent of the server's annotation helper.
_MUTATING_HINTS = {
    "save_strategy": (False, True, False, False),
    "validate_strategy": (False, True, False, True),
    "dry_run_strategy": (False, True, False, True),
    "promote_strategy": (False, True, True, False),
    "ensure_data": (False, True, False, True),
    "start_backtest": (False, True, False, True),
    "start_walk_forward": (False, True, False, True),
    "start_optimization": (False, True, False, True),
    "get_job_status": (False, True, True, False),
    "confirm_optimization": (False, True, True, True),
    "cancel_job": (False, True, True, False),
    "record_feature_request": (False, False, False, False),
}


def _assert_constrained_objects(node: object, location: str) -> None:
    """Allow typed dynamic maps, but never arbitrary dict[str, Any] objects."""
    if isinstance(node, dict):
        if node.get("type") == "object" and "properties" not in node:
            values = node.get("additionalProperties")
            assert isinstance(values, dict) and values, location
        for key, value in node.items():
            _assert_constrained_objects(value, f"{location}/{key}")
    elif isinstance(node, list):
        for index, value in enumerate(node):
            _assert_constrained_objects(value, f"{location}/{index}")


async def test_all_tools_advertise_constrained_output_schemas_and_hints(
    mcp_workspace: MCPWorkspace,
) -> None:
    expected_hints = {
        **dict.fromkeys(_READ_ONLY_TOOLS, (True, False, True, False)),
        **_MUTATING_HINTS,
    }
    malformed_payloads: list[dict[str, object]] = [
        {},
        {"result": {}},
        {"result": [{}]},
    ]
    async with mcp_workspace.connect() as client:
        tools = {tool.name: tool for tool in (await client.session.list_tools()).tools}
        assert len(tools) == 25
        assert tools.keys() == expected_hints.keys() == client.output_schemas.keys()
        for name, tool in tools.items():
            assert tool.description and tool.description.strip(), name
            assert tool.annotations is not None, name
            annotations = tool.annotations
            assert (
                annotations.readOnlyHint,
                annotations.destructiveHint,
                annotations.idempotentHint,
                annotations.openWorldHint,
            ) == expected_hints[name], name

            schema = client.output_schemas[name]
            assert schema == tool.outputSchema
            Draft202012Validator.check_schema(schema)
            assert schema["type"] == "object", name
            assert schema.get("properties"), name
            _assert_constrained_objects(schema, name)
            # A fixed record or a list of fixed records must not degrade to an
            # arbitrary JSON container, including inside the SDK's result wrap.
            for malformed in malformed_payloads:
                with pytest.raises(ValidationError):
                    validate(instance=malformed, schema=schema)


async def test_application_errors_follow_advertised_output_schemas(
    mcp_workspace: MCPWorkspace,
) -> None:
    missing_strategy = {"strategy_id": "missing_strategy"}
    missing_job = {"job_id": "missing_job"}
    run_arguments = {
        **missing_strategy,
        "symbol": "BTC/USDT",
        "timeframe": "1h",
        "start_date": "2024-01-01",
        "end_date": "2024-01-08",
    }
    failures: list[tuple[str, dict[str, Any], str, object]] = [
        (name, missing_strategy, "success", False)
        for name in (
            "get_strategy",
            "validate_strategy",
            "dry_run_strategy",
            "promote_strategy",
        )
    ]
    failures.extend(
        [
            ("get_run", {"run_id": "missing_run"}, "success", False),
            (
                "get_artifact",
                {"artifact_id": "missing_artifact"},
                "success",
                False,
            ),
            (
                "compare_runs",
                {"run_ids": ["missing_a", "missing_b"]},
                "success",
                False,
            ),
            ("get_job_status", missing_job, "status", "not_found"),
            ("cancel_job", missing_job, "success", False),
            ("confirm_optimization", missing_job, "success", False),
            ("start_backtest", run_arguments, "job_id", ""),
            ("start_walk_forward", run_arguments, "job_id", ""),
            (
                "start_optimization",
                {
                    **missing_strategy,
                    "symbol": "BTC/USDT",
                    "timeframe": "1h",
                    "param_ranges": {"fast_period": [3, 5]},
                    "optimization_metric": "sharpe_ratio",
                    "train_start": "2024-01-01",
                    "train_end": "2024-01-03",
                    "test_start": "2024-01-04",
                    "test_end": "2024-01-08",
                },
                "job_id",
                "",
            ),
            (
                "save_strategy",
                {
                    "strategy_id": "invalid_strategy",
                    "code": "def broken(:",
                    "yaml_config": "strategy: {}",
                },
                "success",
                False,
            ),
        ]
    )
    async with mcp_workspace.connect() as client:
        for name, arguments, field, expected in failures:
            # The harness checks the original structuredContent, before
            # unwrapping unions, and rejects MCP isError application replies.
            response = await client.call(name, **arguments)
            assert response[field] == expected, (name, response)
            assert isinstance(response["code"], str) and response["code"], name
            assert response.get("error") or response.get("message"), name
        assert await client.call("list_jobs") == []
        assert await client.call("list_runs") == []


async def test_missing_required_inputs_are_mcp_errors_without_side_effects(
    mcp_workspace: MCPWorkspace,
) -> None:
    async with mcp_workspace.connect() as client:
        tools = (await client.session.list_tools()).tools
        required_tools = [tool for tool in tools if tool.inputSchema.get("required")]
        assert {"save_strategy", "start_backtest", "record_feature_request"} <= {
            tool.name for tool in required_tools
        }
        for tool in required_tools:
            result = await client.session.call_tool(tool.name, {})
            assert result.isError, (tool.name, result)
            assert result.content, tool.name
            assert result.structuredContent is None, tool.name
        assert await client.call("list_jobs") == []
        assert await client.call("list_runs") == []
        assert await client.call("list_feature_requests") == []
        assert not list(mcp_workspace.workspace.rglob("*.py"))
