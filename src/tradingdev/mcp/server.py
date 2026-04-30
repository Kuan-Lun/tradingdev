"""TradingDev FastMCP server."""

from __future__ import annotations

import argparse
from pathlib import Path

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.server import (  # type: ignore[attr-defined]
    TransportSecuritySettings,
)

from tradingdev.app.artifact_service import ArtifactService
from tradingdev.app.capability_service import CapabilityService
from tradingdev.app.data_service import DataService
from tradingdev.app.feature_request_service import FeatureRequestService
from tradingdev.app.job_service import JobService
from tradingdev.app.optimization_service import OptimizationService
from tradingdev.app.run_service import RunService
from tradingdev.app.strategy_service import StrategyService
from tradingdev.mcp.prompts import SERVER_INSTRUCTIONS
from tradingdev.mcp.tools import (
    artifacts,
    backtest,
    data,
    feature_requests,
    jobs,
    optimization,
    runs,
    strategy,
)

_PACKAGE_ROOT = Path(__file__).resolve().parents[1]

mcp = FastMCP(
    name="tradingdev",
    json_response=True,
    transport_security=TransportSecuritySettings(
        enable_dns_rebinding_protection=False,
    ),
    instructions=SERVER_INSTRUCTIONS,
)


def _register_tools() -> None:
    strategy_service = StrategyService()
    data_service = DataService()
    job_service = JobService(
        strategy_service=strategy_service,
        data_service=data_service,
    )
    optimization_service = OptimizationService(strategy_service=strategy_service)
    run_service = RunService()
    artifact_service = ArtifactService(strategy_service=strategy_service)
    feature_request_service = FeatureRequestService()
    capability_service = CapabilityService(feature_request_service)

    strategy.register(mcp, strategy_service, _PACKAGE_ROOT)
    data.register(mcp, data_service)
    backtest.register(mcp, job_service)
    optimization.register(mcp, optimization_service, job_service)
    jobs.register(mcp, job_service)
    runs.register(mcp, run_service)
    artifacts.register(mcp, artifact_service)
    feature_requests.register(mcp, feature_request_service, capability_service)


_register_tools()


def main(argv: list[str] | None = None) -> None:
    """Run the TradingDev MCP server."""
    parser = argparse.ArgumentParser(description="TradingDev MCP Server")
    parser.add_argument(
        "--web",
        action="store_true",
        help="Run as HTTP server. Default: stdio.",
    )
    parser.add_argument(
        "--transport",
        choices=["sse", "streamable-http"],
        default="streamable-http",
        help="HTTP transport when --web is set.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="HTTP port when --web is set.",
    )
    args = parser.parse_args(argv)

    if args.web:
        mcp.settings.port = args.port
        mcp.run(transport=args.transport)
    else:
        mcp.run()


if __name__ == "__main__":
    main()
