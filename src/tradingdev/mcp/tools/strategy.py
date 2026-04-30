"""Strategy lifecycle MCP tools."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from tradingdev.domain.strategies.templates import strategy_contract_payload
from tradingdev.mcp.schemas import SaveStrategyInput, ToolResult

if TYPE_CHECKING:
    from pathlib import Path

    from mcp.server.fastmcp import FastMCP

    from tradingdev.app.strategy_service import StrategyService


def register(mcp: FastMCP, service: StrategyService, package_root: Path) -> None:
    """Register strategy lifecycle tools."""

    @mcp.tool()
    def get_strategy_contract() -> dict[str, str]:
        """Return reference code and YAML contract for generated strategies."""
        return strategy_contract_payload(package_root)

    @mcp.tool()
    def list_strategies() -> list[dict[str, Any]]:
        """List bundled and generated strategies."""
        return service.list_strategies()

    @mcp.tool()
    def get_strategy(strategy_id: str) -> dict[str, Any]:
        """Retrieve source, YAML config, and metadata for a strategy."""
        return service.get_strategy(strategy_id)

    @mcp.tool()
    def save_strategy(
        strategy_id: str,
        code: str,
        yaml_config: str,
        request_summary: str = "",
    ) -> dict[str, Any]:
        """Save generated strategy code and YAML as a draft."""
        payload = SaveStrategyInput(
            strategy_id=strategy_id,
            code=code,
            yaml_config=yaml_config,
            request_summary=request_summary,
        )
        saved = service.save_draft(
            payload.strategy_id,
            payload.code,
            payload.yaml_config,
            request_summary=payload.request_summary,
        )
        result = ToolResult(
            success=saved.success,
            message="Draft strategy saved." if saved.success else "",
            error=saved.error,
        )
        return {
            **result.model_dump(mode="json"),
            "strategy_id": saved.strategy_id,
            "py_path": saved.source_path,
            "yaml_path": saved.config_path,
            "status": saved.status,
        }

    @mcp.tool()
    def validate_strategy(strategy_id: str) -> dict[str, Any]:
        """Validate a generated strategy draft.

        This currently executes generated Python strategy code during the smoke
        contract check. Sandboxed execution isolation is future work.
        """
        return service.validate(strategy_id)

    @mcp.tool()
    def dry_run_strategy(strategy_id: str) -> dict[str, Any]:
        """Run a signal-generation dry run for a validated strategy.

        This currently executes generated Python strategy code. Sandboxed
        execution isolation is future work.
        """
        return service.dry_run(strategy_id)
