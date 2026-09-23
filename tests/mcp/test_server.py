"""Server composition and import side-effect tests."""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

import pytest

from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.mcp.server import create_server

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture
def runtime_root(tmp_path: Path) -> Iterator[Path]:
    with TemporaryDirectory(dir=tmp_path) as temporary:
        yield Path(temporary)


def test_import_and_help_do_not_create_a_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    env = {**os.environ, "TRADINGDEV_WORKSPACE": str(workspace)}
    for args in (
        ["-c", "import tradingdev.mcp.server"],
        ["-m", "tradingdev.mcp.server", "--help"],
    ):
        result = subprocess.run(  # noqa: S603
            [sys.executable, *args],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        assert not workspace.exists()


def test_server_instances_keep_generated_strategies_in_their_workspace(
    runtime_root: Path,
) -> None:
    first_workspace = WorkspacePaths(runtime_root / "first")
    second_workspace = WorkspacePaths(runtime_root / "second")
    first = create_server(first_workspace)
    second = create_server(second_workspace)

    async def check() -> None:
        saved_result = await first.call_tool(
            "save_strategy",
            {
                "strategy_id": "isolated_strategy",
                "code": "class IsolatedStrategy: pass\n",
                "yaml_config": (
                    "strategy:\n  id: isolated_strategy\n"
                    "  class_name: IsolatedStrategy\n"
                ),
            },
        )
        assert isinstance(saved_result, tuple)
        assert isinstance(saved_result[1], dict)
        assert saved_result[1]["result"]["success"] is True
        first_result = await first.call_tool(
            "get_strategy", {"strategy_id": "isolated_strategy"}
        )
        second_result = await second.call_tool(
            "get_strategy", {"strategy_id": "isolated_strategy"}
        )
        assert isinstance(first_result, tuple)
        assert isinstance(first_result[1], dict)
        assert first_result[1]["result"]["success"] is True
        assert isinstance(second_result, tuple)
        assert isinstance(second_result[1], dict)
        assert second_result[1]["result"]["success"] is False

    asyncio.run(check())
    assert list(
        (first_workspace.generated_strategies / "isolated_strategy" / "revisions").glob(
            "*/strategy.py"
        )
    )
    assert not (second_workspace.generated_strategies / "isolated_strategy").exists()


def test_revision_selectors_are_advertised_in_tool_schemas(runtime_root: Path) -> None:
    server = create_server(WorkspacePaths(runtime_root / "workspace"))
    tools = {tool.name: tool for tool in asyncio.run(server.list_tools())}
    for name in (
        "get_strategy",
        "validate_strategy",
        "dry_run_strategy",
        "promote_strategy",
        "start_backtest",
        "start_walk_forward",
        "start_optimization",
    ):
        schema = tools[name].inputSchema
        assert "revision_id" in schema["properties"]
        assert "revision_id" not in schema.get("required", [])
    saved_schema = tools["save_strategy"].outputSchema
    assert saved_schema is not None
    assert "revision_id" in saved_schema["$defs"]["StrategySaveSuccess"]["required"]
