"""Explicit representative choices and immutable historical snapshots."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, StringConstraints

from tradingdev.domain.execution import ExecutionManifest
from tradingdev.domain.performance.artifacts import (
    MetricDefinitionSnapshot,
    PerformanceScope,
)

MAX_REPRESENTATIVE_BYTES = 131072
type RepresentativeId = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{32}$")]
type RepresentativeVersion = Annotated[int, Field(strict=True, ge=1)]
type ResearchGroup = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=120)
]
type SelectionReason = Annotated[str, StringConstraints(max_length=2000)]
type SavedRunId = Annotated[
    str, StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")
]
type SavedScope = Annotated[str, StringConstraints(min_length=1, max_length=128)]
type Digest = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]


class RepresentativeModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid", allow_inf_nan=False, revalidate_instances="always"
    )


class RepresentativeRef(RepresentativeModel):
    representative_id: RepresentativeId
    version: RepresentativeVersion


class RepresentativeSnapshot(RepresentativeModel):
    run_id: SavedRunId
    scope: SavedScope
    strategy_id: str = Field(min_length=1)
    revision_id: str | None
    source_hash: Digest
    performance_sha256: Digest
    observations_sha256: Digest
    manifest: ExecutionManifest
    effective_parameters: dict[str, JsonValue]
    performance: PerformanceScope
    definitions: dict[str, MetricDefinitionSnapshot]


class RepresentativeRecord(RepresentativeRef):
    schema_version: Literal[1] = 1
    group_id: ResearchGroup
    created_at: str
    selected_at: str
    reason: SelectionReason = ""
    snapshot: RepresentativeSnapshot


class RepresentativeConflictError(ValueError):
    """A duplicate choice or a stale expected version cannot replace a selection."""
