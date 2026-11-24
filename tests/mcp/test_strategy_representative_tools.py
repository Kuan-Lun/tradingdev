"""Representative lifecycle and pinned rankings through registered MCP tools."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import pytest
from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp_types import CallToolResult
from tests.representative_fixtures import publish_representative_run

from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.adapters.storage.sqlite import SQLiteStore
from tradingdev.app.ml_service import MLService
from tradingdev.app.representative_service import RepresentativeService
from tradingdev.mcp.strict_server import StrictMCPServer
from tradingdev.mcp.tools import ml, representatives

if TYPE_CHECKING:
    from pathlib import Path


def call(server: MCPServer, name: str, **arguments: Any) -> dict[str, Any]:
    async def run() -> dict[str, Any]:
        result = await server.call_tool(name, arguments)
        assert isinstance(result, CallToolResult)
        if result.is_error:
            return {"success": False, "tool_error": True}
        assert result.structured_content is not None
        payload = result.structured_content
        value = payload.get("result", payload)
        assert isinstance(value, dict)
        return value

    return asyncio.run(run())


@pytest.fixture(params=[MCPServer, StrictMCPServer], ids=["sdk", "strict"])
def runtime(
    tmp_path: Path, request: pytest.FixtureRequest
) -> tuple[MCPServer, WorkspacePaths, SQLiteStore]:
    workspace = WorkspacePaths(tmp_path / "workspace")
    store = SQLiteStore(workspace)
    server = request.param("test-representatives")
    ml.register(server, MLService(workspace=workspace))
    representatives.register(server, RepresentativeService(workspace, store))
    return server, workspace, store


def test_discovery_explicit_selection_new_experiment_revision_and_stale_writer(
    runtime: tuple[MCPServer, WorkspacePaths, SQLiteStore],
) -> None:
    server, workspace, store = runtime
    contract = call(server, "get_ml_contract")
    discovery = contract["ranking"]["representatives"]
    assert discovery["create_tool"] == "create_strategy_representative"
    assert discovery["read_tool"] == "get_strategy_representatives"
    assert discovery["revise_tool"] == "revise_strategy_representative"
    assert "latest" in contract["guidance"]["representatives"]
    publish_representative_run(workspace, store, "A", value=1.0)
    saved = call(
        server, discovery["create_tool"], group_id="group", run_id="A", scope="full"
    )
    assert saved["success"] is True and saved["version"] == 1
    ref = {"representative_id": saved["representative_id"], "version": 1}
    publish_representative_run(
        workspace, store, "B", window=20, revision="revision-2", value=9.0
    )
    ranked = call(
        server, "rank_candidates", metric="run.sharpe_ratio", representatives=[ref]
    )
    assert ranked["success"] is True
    row = ranked["ranked"][0]
    assert (row["run_id"], row["revision_id"], row["value"]) == ("A", "revision-1", 1.0)
    assert row["parameters"] == {"window": 10, "nested": {"default": 7}}
    assert row["representative"]["version"] == 1
    assert any("holdout" in text for text in ranked["caveats"])
    updated = call(
        server,
        discovery["revise_tool"],
        representative_id=ref["representative_id"],
        expected_version=1,
        run_id="B",
        scope="full",
    )
    assert updated["version"] == 2
    assert (
        call(server, discovery["read_tool"], **ref)["representative"]["snapshot"][
            "run_id"
        ]
        == "A"
    )
    pinned = call(
        server, "rank_candidates", metric="run.sharpe_ratio", representatives=[ref]
    )
    assert pinned["ranked"][0]["run_id"] == "A"
    ref["version"] = 2
    assert (
        call(
            server, "rank_candidates", metric="run.sharpe_ratio", representatives=[ref]
        )["ranked"][0]["run_id"]
        == "B"
    )
    stale = call(
        server,
        discovery["revise_tool"],
        representative_id=ref["representative_id"],
        expected_version=1,
        run_id="A",
        scope="full",
    )
    assert stale["code"] == "representative_conflict"
    assert (
        call(
            server, discovery["read_tool"], representative_id=ref["representative_id"]
        )["representative"]["version"]
        == 2
    )


def test_skipped_representatives_keep_conditions_missing_reasons_and_source_identity(
    runtime: tuple[MCPServer, WorkspacePaths, SQLiteStore],
) -> None:
    server, workspace, store = runtime
    refs = []
    for run_id, strategy_id, value, fees, timeframe in (
        ("A", "one", 1.0, 0.001, "1d"),
        ("missing-value", "two", None, 0.002, "1h"),
        ("lost", "three", 20.0, 0.003, "2h"),
    ):
        publish_representative_run(
            workspace,
            store,
            run_id,
            strategy_id=strategy_id,
            value=value,
            fees=fees,
            timeframe=timeframe,
            start_date="2025-01-01" if run_id == "missing-value" else "2024-01-01",
            end_date="2025-03-01" if run_id == "missing-value" else "2024-03-01",
        )
        selected = call(
            server,
            "create_strategy_representative",
            group_id="group",
            run_id=run_id,
            scope="full",
        )
        assert selected["success"] is True
        refs.append({"representative_id": selected["representative_id"], "version": 1})
    (workspace.runs / "lost" / "performance.json").unlink()
    result = call(
        server, "rank_candidates", metric="run.sharpe_ratio", representatives=refs
    )
    assert result["success"] is True
    assert (
        result["n_candidates"] == 3
        and result["n_ranked"] == 1
        and result["n_skipped"] == 2
    )
    assert result["ranked"][0]["run_id"] == "A"
    skipped = {r["run_id"]: r for r in result["skipped"]}
    assert skipped["missing-value"]["metric_unavailable_reason"] == "zero_denominator"
    assert skipped["missing-value"]["source_available"] is True
    assert skipped["missing-value"]["conditions"]["timeframe"] == "1h"
    assert skipped["missing-value"]["conditions"]["start_date"].startswith("2025-01-01")
    assert result["ranked"][0]["conditions"]["start_date"].startswith("2024-01-01")
    assert skipped["lost"]["source_available"] is False
    assert skipped["lost"]["conditions"]["fees"] == 0.003
    assert skipped["lost"]["representative"]["performance_sha256"]
    assert skipped["lost"]["reason"]
    read = call(server, "get_strategy_representatives", **refs[-1])
    assert read["source_available"] is False
    assert read["representative"]["snapshot"]["run_id"] == "lost"


def test_duplicate_versions_aliases_empty_selection_and_source_mixing_are_refused(
    runtime: tuple[MCPServer, WorkspacePaths, SQLiteStore],
) -> None:
    server, workspace, store = runtime
    publish_representative_run(workspace, store, "A")
    refs = []
    for group in ("one", "two"):
        saved = call(
            server,
            "create_strategy_representative",
            group_id=group,
            run_id="A",
            scope="full",
        )
        refs.append({"representative_id": saved["representative_id"], "version": 1})
    for selection in ([refs[0], refs[0]], refs):
        assert (
            call(
                server,
                "rank_candidates",
                metric="run.sharpe_ratio",
                representatives=selection,
            )["code"]
            == "duplicate_selections"
        )
    for arguments in (
        {"metric": "signal.ic_mean", "representatives": refs[:1]},
        {
            "metric": "run.sharpe_ratio",
            "runs": [{"run_id": "A", "scope": "full"}],
            "representatives": refs[:1],
        },
        {
            "metric": "run.sharpe_ratio",
            "representatives": refs[:1],
            "symbol": "BTC/USDT",
        },
    ):
        assert call(server, "rank_candidates", **arguments)["success"] is False
    with pytest.raises(ToolError, match="at least 1"):
        call(server, "rank_candidates", metric="run.sharpe_ratio", representatives=[])


def test_strict_mcp_rejects_forged_parameters_revision_and_unknown_selection_fields(
    tmp_path: Path,
) -> None:
    workspace = WorkspacePaths(tmp_path / "workspace")
    store = SQLiteStore(workspace)
    server = StrictMCPServer("forged-representatives")
    service = RepresentativeService(workspace, store)
    representatives.register(server, service)
    publish_representative_run(workspace, store, "A")
    for forged in ({"parameters": {"window": 999}}, {"revision_id": "forged"}):
        with pytest.raises(ToolError, match="Unknown arguments"):
            call(
                server,
                "create_strategy_representative",
                group_id="group",
                run_id="A",
                scope="full",
                **forged,
            )
        assert service.get()["total"] == 0
