"""Bounded MCP projections of explicit representative choices."""

from typing import Literal

from tradingdev.app.contracts.common import ContractModel
from tradingdev.domain.representatives import (
    RepresentativeId,
    RepresentativeRecord,
    RepresentativeVersion,
    ResearchGroup,
    SavedRunId,
    SavedScope,
)


class RepresentativeSaved(ContractModel):
    success: Literal[True]
    representative_id: RepresentativeId
    version: RepresentativeVersion


class RepresentativeRead(ContractModel):
    success: Literal[True]
    representative: RepresentativeRecord
    source_available: bool
    unavailable_reason: str | None


class RepresentativeSummary(ContractModel):
    representative_id: RepresentativeId
    version: RepresentativeVersion
    group_id: ResearchGroup
    strategy_id: str
    run_id: SavedRunId
    scope: SavedScope
    revision_id: str | None
    source_available: bool
    unavailable_reason: str | None


class RepresentativeList(ContractModel):
    success: Literal[True]
    total: int
    offset: int
    limit: int
    next_offset: int | None
    items: list[RepresentativeSummary]
