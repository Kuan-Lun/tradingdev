"""Real stdio MCP persistence, restart, pinned ranking and missing-source reads."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from tests.representative_fixtures import publish_representative_run
from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.adapters.storage.sqlite import SQLiteStore

if TYPE_CHECKING:
    from tests.integration.mcp_harness import MCPWorkspace

pytestmark = [pytest.mark.integration, pytest.mark.anyio]


async def test_representative_choices_survive_mcp_restart_without_reselection(
    mcp_workspace: MCPWorkspace,
) -> None:
    paths = WorkspacePaths(mcp_workspace.workspace)
    store = SQLiteStore(paths)
    publish_representative_run(paths, store, "A", value=0.5)
    async with mcp_workspace.connect() as client:
        saved = await client.call(
            "create_strategy_representative", group_id="group", run_id="A", scope="full"
        )
        identity = saved["representative_id"]
        publish_representative_run(
            paths, store, "B", window=20, revision="revision-2", value=9.0
        )
        current = await client.call(
            "get_strategy_representatives", representative_id=identity
        )
        assert current["representative"]["snapshot"]["run_id"] == "A"
        changed = await client.call(
            "revise_strategy_representative",
            representative_id=identity,
            expected_version=1,
            run_id="B",
            scope="full",
        )
        assert changed["version"] == 2
    async with mcp_workspace.connect() as client:
        current = await client.call(
            "get_strategy_representatives", representative_id=identity
        )
        assert current["representative"]["version"] == 2
        assert current["representative"]["snapshot"]["run_id"] == "B"
        old = {"representative_id": identity, "version": 1}
        ranked = await client.call(
            "rank_candidates", metric="run.sharpe_ratio", representatives=[old]
        )
        assert ranked["ranked"][0]["run_id"] == "A"
        assert ranked["ranked"][0]["value"] == 0.5
        stale = await client.call(
            "revise_strategy_representative",
            representative_id=identity,
            expected_version=1,
            run_id="A",
            scope="full",
        )
        assert stale["code"] == "representative_conflict"
        (paths.runs / "A" / "performance.json").unlink()
        read = await client.call("get_strategy_representatives", **old)
        assert read["source_available"] is False
        assert read["representative"]["snapshot"]["run_id"] == "A"
        result = await client.call(
            "rank_candidates", metric="run.sharpe_ratio", representatives=[old]
        )
        assert result["n_ranked"] == 0 and result["n_skipped"] == 1
        assert result["skipped"][0]["representative"]["version"] == 1
        assert result["skipped"][0]["source_available"] is False
    assert not store.list_jobs()
