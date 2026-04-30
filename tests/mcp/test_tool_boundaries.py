"""MCP tool boundary tests."""

from pathlib import Path


def test_promote_strategy_is_registered_with_artifact_tools() -> None:
    strategy_tools = Path("src/tradingdev/mcp/tools/strategy.py").read_text(
        encoding="utf-8"
    )
    artifact_tools = Path("src/tradingdev/mcp/tools/artifacts.py").read_text(
        encoding="utf-8"
    )

    assert "def promote_strategy" not in strategy_tools
    assert "def promote_strategy" in artifact_tools


def test_job_tools_include_cancel_job() -> None:
    job_tools = Path("src/tradingdev/mcp/tools/jobs.py").read_text(encoding="utf-8")

    assert "def cancel_job" in job_tools


def test_dashboard_uses_run_and_artifact_services() -> None:
    dashboard = Path("src/tradingdev/adapters/dashboard/app.py").read_text(
        encoding="utf-8"
    )

    assert "RunService" in dashboard
    assert "ArtifactService" in dashboard
    assert "DataManager" not in dashboard
    assert "load_cached_result" not in dashboard
    assert "load_config" not in dashboard
