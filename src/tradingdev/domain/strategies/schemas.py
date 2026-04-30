"""Domain schemas for generated strategy lifecycle state."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class StrategyStatus(StrEnum):
    """Allowed lifecycle states for generated strategies."""

    DRAFT = "draft"
    VALIDATED = "validated"
    RUNNABLE = "runnable"
    PROMOTED = "promoted"
    REJECTED = "rejected"


class StrategyDiagnostic(BaseModel):
    """One static or runtime validation diagnostic."""

    level: Literal["error", "warning"] = "error"
    code: str
    phase: str
    message: str
    line: int | None = None
    fix: str | None = None


class ValidationResult(BaseModel):
    """Structured validation result persisted with strategy metadata."""

    checked_at: str
    success: bool
    diagnostics: list[StrategyDiagnostic] = Field(default_factory=list)
    signal_analysis: dict[str, Any] = Field(default_factory=dict)

    @property
    def has_error(self) -> bool:
        """Return whether any diagnostic is an error."""
        return any(item.level == "error" for item in self.diagnostics)


class StrategyMetadata(BaseModel):
    """Persisted metadata for a generated strategy artifact."""

    model_config = ConfigDict(extra="allow")

    strategy_id: str
    class_name: str
    artifact_type: str = "generated_strategy"
    status: StrategyStatus
    created_at: str
    updated_at: str
    request_summary: str = ""
    source_path: str
    config_path: str
    source_hash: str
    config_hash: str
    validation: ValidationResult | None = None
    dry_run: ValidationResult | None = None


class StrategySpec(BaseModel):
    """Runtime strategy source and config pointer."""

    strategy_id: str
    class_name: str
    source_path: str
    config_path: str
    status: StrategyStatus
    kind: Literal["generated", "bundled"] = "generated"
    metadata: StrategyMetadata | dict[str, Any] | None = None
