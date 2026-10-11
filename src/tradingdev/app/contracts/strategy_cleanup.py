"""Explicit preview and apply responses for generated draft cleanup."""

from typing import Literal

from tradingdev.app.contracts.common import ContractModel


class StrategyCleanupItem(ContractModel):
    """One revision's eligibility, protection, or actual deletion outcome."""

    revision_id: str
    outcome: Literal["eligible", "protected", "deleted", "failed", "missing"]
    reasons: list[str]


class StrategyCleanupResult(ContractModel):
    """Cleanup never infers permission to delete from a preview."""

    success: bool
    strategy_id: str
    applied: bool
    revisions: list[StrategyCleanupItem]
