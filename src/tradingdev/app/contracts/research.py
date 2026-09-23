"""Typed read contracts for runs, artifacts, and feature requests."""

from typing import Literal

from pydantic import JsonValue

from tradingdev.app.contracts.common import ContractModel


class RunRecord(ContractModel):
    """Persisted run identity, execution lineage, and dynamic metrics."""

    run_id: str
    job_id: str
    strategy_id: str
    config_hash: str | None
    source_hash: str | None
    random_seed: int | None
    dataset_id: str | None
    metrics: dict[str, JsonValue]
    artifact_dir: str
    created_at: str


class RunResponse(ContractModel):
    """Successful lookup of a completed run."""

    success: Literal[True]
    run: RunRecord


class RunComparison(ContractModel):
    """Selected metrics for one run in a comparison."""

    run_id: str
    strategy_id: str
    metrics: dict[str, JsonValue]


class CompareRunsResponse(ContractModel):
    """Successful comparison of the selected completed runs."""

    success: Literal[True]
    runs: list[RunComparison]


class ArtifactIdentity(ContractModel):
    """Persisted artifact identity and file location."""

    artifact_id: str
    run_id: str | None
    artifact_type: str
    path: str
    sha256: str | None
    created_at: str


class ArtifactRecord(ArtifactIdentity):
    """Artifact identity with metadata supplied by its producer."""

    metadata: dict[str, JsonValue]


class ArtifactResponse(ContractModel):
    """Successful artifact lookup with optional UTF-8 content."""

    success: Literal[True]
    artifact: ArtifactRecord
    content: str | None = None


class FeatureRequestMetadata(ContractModel):
    """The feature request recorded in an artifact's metadata."""

    request_id: str
    title: str
    description: str
    source_tool: str
    created_at: str
    status: Literal["open"]


class FeatureRequestArtifact(ArtifactIdentity):
    """An artifact with the fixed feature-request metadata contract."""

    artifact_type: Literal["feature_request"]
    metadata: FeatureRequestMetadata


class RecordedFeatureRequest(ContractModel):
    """Location of the newly recorded feature request."""

    success: Literal[True]
    request_id: str
    path: str


class RecordFeatureRequestResponse(ContractModel):
    """Successful creation of a local feature request artifact."""

    success: Literal[True]
    message: str
    feature_request: RecordedFeatureRequest
