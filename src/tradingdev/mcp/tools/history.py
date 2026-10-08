"""Saved-run discovery, paginated observations, and standard research reports."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from mcp.types import ToolAnnotations

from tradingdev.app.contracts.common import ErrorResponse
from tradingdev.app.contracts.history import (
    FindRunsResponse,
    HistoryQueryError,
    RunEquityResponse,
    RunTradesResponse,
)
from tradingdev.app.contracts.reports import (
    ReportCommentary,
    ReportResponse,
    ReportSectionCatalog,
)

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

    from tradingdev.app.report_service import ReportService
    from tradingdev.app.trade_history_service import TradeHistoryService


def register(
    mcp: FastMCP, history: TradeHistoryService, reports: ReportService
) -> None:
    """Keep all clients on the same verified historical data services."""
    read_only = ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    )

    @mcp.tool(annotations=read_only)
    def find_runs(
        strategy_id: str | None = None,
        parameters: dict[str, Any] | None = None,
        symbol: str | None = None,
        timeframe: str | None = None,
        offset: int = 0,
        limit: int = 50,
    ) -> FindRunsResponse | HistoryQueryError:
        """Find saved run/scopes by historical effective parameter subsets.

        Each optimization trial is a separate match. Select a run_id and scope
        before requesting trades; the same parameters can have different dates,
        costs, revisions, and datasets. limit is 1..500. No strategy is rerun.
        """
        result = history.find_runs(
            strategy_id=strategy_id,
            parameters=parameters,
            symbol=symbol,
            timeframe=timeframe,
            offset=offset,
            limit=limit,
        )
        if result.get("success") is False:
            return HistoryQueryError.model_validate(result)
        return FindRunsResponse.model_validate(result)

    @mcp.tool(annotations=read_only)
    def get_run_trades(
        run_id: str,
        scope: str | None = None,
        offset: int = 0,
        limit: int = 50,
        status: Literal["open", "closed"] | None = None,
        direction: Literal["long", "short"] | None = None,
        entry_start: str | None = None,
        entry_end: str | None = None,
    ) -> RunTradesResponse | HistoryQueryError:
        """Read saved position/trade records with UTC dates and stable pagination.

        Open positions have a final mark, not an executed exit. These are paired
        trades, not individual orders. Filter dates refer to entry time; date-only
        end includes that entire UTC day, timestamps are inclusive instants.
        Omitted scope selects the saved default; aggregate fold scopes have no
        individual trades. limit is 1..500. Never recalculates or reruns a strategy.
        """
        result = history.get_run_trades(
            run_id,
            scope=scope,
            offset=offset,
            limit=limit,
            status=status,
            direction=direction,
            entry_start=entry_start,
            entry_end=entry_end,
        )
        if result.get("success") is False:
            return HistoryQueryError.model_validate(result)
        return RunTradesResponse.model_validate(result)

    @mcp.tool(annotations=read_only)
    def get_run_equity(
        run_id: str,
        scope: str | None = None,
        offset: int = 0,
        limit: int = 100,
        start: str | None = None,
        end: str | None = None,
    ) -> RunEquityResponse | HistoryQueryError:
        """Read aligned saved per-bar equity or cumulative volume-mode PnL.

        Date-only end includes that entire UTC day; timestamps are inclusive
        instants. limit is 1..500. Volume mode has no initial capital and its
        observed values must not be described as account balances.
        """
        result = history.get_run_equity(
            run_id, scope=scope, offset=offset, limit=limit, start=start, end=end
        )
        if result.get("success") is False:
            return HistoryQueryError.model_validate(result)
        return RunEquityResponse.model_validate(result)

    @mcp.tool(annotations=read_only)
    def get_report_sections() -> ReportSectionCatalog:
        """Discover backend-rendered sections and suggested report recipes.

        The client chooses sections and writes interpretation; the server renders
        the HTML, tables, and charts. Recipes are optional, not enforced layouts.
        """
        return ReportSectionCatalog.model_validate(reports.get_report_sections())

    @mcp.tool(
        annotations=ToolAnnotations(
            readOnlyHint=False,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=False,
        )
    )
    def generate_report(
        run_ids: list[str],
        sections: list[str] | None = None,
        commentary: list[ReportCommentary] | None = None,
    ) -> ReportResponse | ErrorResponse:
        """Compose an offline HTML report from 1..8 saved runs and client prose.

        Discover section IDs/recipes with get_report_sections. Omit sections for
        the standard recipe, provide IDs in desired order, or [] to render no
        built-in sections while keeping source identity and supplied commentary.
        Every report embeds the complete loaded data for all scopes of the runs,
        including saved trades, equity and execution configuration, even with [].
        Section selection does not remove underlying data from the HTML.
        Commentary contains plain-text title/text objects (at most 20, total
        20,000 characters); HTML is escaped and text is
        labelled client interpretation, distinct from recorded calculations.
        Selected trade tables contain all records with local CSV download.
        Returns a registered artifact and local path. Existing backtests remain
        unchanged. No strategy, model, external website, or market download runs.
        """
        result = reports.generate_report(
            run_ids,
            sections=sections,
            commentary=[item.model_dump() for item in commentary]
            if commentary is not None
            else None,
        )
        if result.get("success") is False:
            return ErrorResponse.model_validate(result)
        return ReportResponse.model_validate(result)
