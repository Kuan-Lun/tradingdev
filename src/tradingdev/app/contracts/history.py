"""Typed, bounded queries of original backtest observations."""

from typing import Literal

from pydantic import Field, JsonValue

from tradingdev.app.contracts.common import ContractModel, ErrorResponse


class HistoryQueryError(ErrorResponse):
    """A failed history selection, with usable scalar scopes when available."""

    available_scopes: list[str] = Field(default_factory=list)


class HistoryIssue(ContractModel):
    """A run that could not be included in a discovery query."""

    run_id: str
    code: str
    error: str


class HistoryRunMatch(ContractModel):
    """One parameterized scalar scope, not an entire optimization search."""

    run_id: str
    strategy_id: str
    revision_id: str | None
    manifest_hash: str | None
    created_at: str
    scope: str
    mode: Literal["signal", "volume"]
    split: Literal["full", "train", "test"] | None
    symbol: str | None
    timeframe: str | None
    parameters: dict[str, JsonValue]
    parameter_provenance: str
    parameters_complete: bool
    selected: bool
    bar_count: int
    trade_count: int


class HistoryPage(ContractModel):
    """Counts refer to the full saved sequence before paging."""

    success: Literal[True]
    offset: int
    limit: int
    total: int
    matched: int
    next_offset: int | None


class FindRunsResponse(HistoryPage):
    """Discovery remains useful while explicitly reporting unreadable runs."""

    runs: list[HistoryRunMatch]
    complete: bool
    issues: list[HistoryIssue]


class HistoryScopeIdentity(HistoryPage):
    """Execution identity and the meaning of a selected stored sequence."""

    run_id: str
    scope: str
    manifest_hash: str | None
    mode: Literal["signal", "volume"]
    parameters: dict[str, JsonValue]
    parameter_provenance: str
    parameters_complete: bool
    calendar_timezone: Literal["UTC"] = "UTC"


class HistoryTrade(ContractModel):
    """A normalized view retains its unmodified source record for audit."""

    trade_id: int
    direction: Literal["long", "short"] | None
    status: Literal["open", "closed"] | None
    entry_timestamp: str | None
    exit_timestamp: str | None
    mark_timestamp: str | None
    entry_price: float | None
    exit_price: float | None
    mark_price: float | None
    size: float | None
    size_quote: float | None
    entry_fees: float | None
    exit_fees: float | None
    fee: float | None
    gross_pnl: float | None
    net_pnl: float | None
    record: dict[str, JsonValue]


class RunTradesResponse(HistoryScopeIdentity):
    """Saved trades; marks of open trades are never presented as exits."""

    trades: list[HistoryTrade]
    date_filter_basis: Literal["entry_timestamp"] = "entry_timestamp"


class HistoryEquityPoint(ContractModel):
    """One original bar, without interpolation or regenerated market data."""

    bar_index: int
    timestamp: str | None
    equity: float | None
    bar_return: float | None


class RunEquityResponse(HistoryScopeIdentity):
    """Account value in signal mode; cumulative PnL in volume mode."""

    init_cash: float | None
    equity_basis: Literal["account_equity", "cumulative_pnl"]
    points: list[HistoryEquityPoint]
