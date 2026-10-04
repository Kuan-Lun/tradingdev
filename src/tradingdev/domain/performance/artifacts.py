"""Versioned, JSON-native performance scopes and original observations."""

from __future__ import annotations

from dataclasses import asdict
from typing import TYPE_CHECKING, Any, Literal, Self

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from tradingdev.domain.performance.catalog import METRIC_CATALOG
from tradingdev.shared.utils.json_values import (
    normalize_json_object,
    normalize_json_value,
)

if TYPE_CHECKING:
    from tradingdev.domain.backtest.pipeline_result import PipelineResult
    from tradingdev.domain.backtest.result import BacktestResult


type MetricScalar = int | float | None


class ArtifactModel(BaseModel):
    """Validate stored structure without coercing booleans or accepting NaN."""

    model_config = ConfigDict(
        extra="forbid", strict=True, frozen=True, allow_inf_nan=False
    )


class FoldStats(ArtifactModel):
    """Descriptive finite test-fold observations, never a pooled return metric."""

    mean: float | None
    std: float | None
    min: float | None
    max: float | None
    valid_count: int = Field(ge=0)


class MetricDefinitionSnapshot(ArtifactModel):
    """The exact metric definition recorded by the result producer."""

    id: str
    unit: str
    category: str
    provider: str
    description: str
    summary: bool
    optimization_direction: Literal["maximize", "minimize"] | None
    modes: tuple[str, ...]
    requires_annualization: bool


class PerformanceScope(ArtifactModel):
    """Complete values and provenance for one split or fold summary."""

    kind: Literal["backtest", "fold_summary"] = "backtest"
    mode: Literal["signal", "volume"]
    values: dict[str, MetricScalar | FoldStats]
    metadata: dict[str, JsonValue]
    split: Literal["full", "train", "test"] | None = "full"
    fold_index: int | None = Field(default=None, ge=0)
    trial_index: int | None = Field(default=None, ge=0)
    parameters: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def value_shape(self) -> Self:
        """Keep scalar results separate from descriptive fold statistics."""
        if self.kind == "backtest" and any(
            isinstance(value, FoldStats) for value in self.values.values()
        ):
            raise ValueError("Backtest scopes require scalar metrics")
        if self.kind == "fold_summary" and any(
            not isinstance(value, FoldStats) for value in self.values.values()
        ):
            raise ValueError("Fold summary scopes require FoldStats values")
        return self


class PerformanceBundle(ArtifactModel):
    """Complete scoped metrics, definitions and their execution identity."""

    schema_version: Literal[1] = 1
    run_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")
    manifest_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    default_scope: str
    scopes: dict[str, PerformanceScope]
    definitions: dict[str, MetricDefinitionSnapshot]
    selected_train_scope: str | None = None

    @model_validator(mode="after")
    def consistent_scopes(self) -> Self:
        """Every advertised scope and definition must resolve without guessing."""
        if self.default_scope not in self.scopes:
            raise ValueError("Default performance scope does not exist")
        if self.selected_train_scope is not None:
            selected = self.scopes.get(self.selected_train_scope)
            if selected is None or selected.split != "train":
                raise ValueError("Selected train scope must reference a train result")
        for key, definition in self.definitions.items():
            if key != definition.id:
                raise ValueError("Metric definition ID differs from its mapping key")
        for scope in self.scopes.values():
            if missing := scope.values.keys() - self.definitions.keys():
                raise ValueError(
                    f"Missing metric definitions: {', '.join(sorted(missing))}"
                )
        return self


class ScopeObservations(ArtifactModel):
    """Aligned original bars and trade records, before presentation selection."""

    init_cash: float | None
    equity_curve: list[float | None]
    returns: list[float | None] | None
    timestamps: list[str] | None
    trades: list[dict[str, JsonValue]]

    @model_validator(mode="after")
    def aligned_observations(self) -> Self:
        """Reject shifted series and references outside the stored bar sequence."""
        count = len(self.equity_curve)
        if self.returns is not None and len(self.returns) != count:
            raise ValueError("Returns and equity observations have different lengths")
        if self.timestamps is not None:
            if len(self.timestamps) != count:
                raise ValueError(
                    "Timestamps and equity observations have different lengths"
                )
            parsed = pd.to_datetime(self.timestamps, utc=True, errors="raise")
            if (
                parsed.hasnans
                or not parsed.is_monotonic_increasing
                or parsed.has_duplicates
            ):
                raise ValueError(
                    "Observation timestamps must be valid and strictly ordered"
                )
        for trade in self.trades:
            for key in ("entry_idx", "exit_idx"):
                if key in trade:
                    index = trade[key]
                    if (
                        isinstance(index, bool)
                        or not isinstance(index, int)
                        or not 0 <= index < count
                    ):
                        raise ValueError(
                            f"Trade {key} lies outside the observation sequence"
                        )
        return self


class ObservationsBundle(ArtifactModel):
    """Original observations for the scalar performance scopes."""

    schema_version: Literal[1] = 1
    run_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")
    manifest_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    scopes: dict[str, ScopeObservations]


class PerformanceArtifacts(ArtifactModel):
    """One validated pair of performance and observations JSON artifacts."""

    performance: PerformanceBundle
    observations: ObservationsBundle

    @model_validator(mode="after")
    def matching_identity_and_scopes(self) -> Self:
        """The two files describe the same run and all scalar scopes exactly once."""
        if (
            self.performance.run_id != self.observations.run_id
            or self.performance.manifest_hash != self.observations.manifest_hash
        ):
            raise ValueError("Performance and observations execution identities differ")
        expected = {
            key
            for key, scope in self.performance.scopes.items()
            if scope.kind == "backtest"
        }
        if expected != self.observations.scopes.keys():
            raise ValueError("Performance and observations scopes differ")
        return self


def _timestamps(result: BacktestResult) -> list[str] | None:
    if result.timestamps is None:
        return None
    values = result.timestamps
    if values.dtype.kind in {"i", "u", "f", "b"}:
        raise ValueError("Numeric observation indices are not timestamps")
    converted = pd.to_datetime(values, utc=True, errors="raise")
    if converted.hasnans:
        raise ValueError("Missing observation timestamp")
    return [timestamp.isoformat() for timestamp in converted]


def scope_from_backtest(
    result: BacktestResult,
    *,
    split: Literal["full", "train", "test"] = "full",
    fold_index: int | None = None,
    trial_index: int | None = None,
    parameters: dict[str, Any] | None = None,
) -> tuple[PerformanceScope, ScopeObservations]:
    """Copy calculated values and observations without recalculating metrics."""
    observations = ScopeObservations.model_validate(
        {
            "init_cash": result.init_cash,
            "equity_curve": normalize_json_value(result.equity_curve.tolist()),
            "returns": normalize_json_value(result.returns.tolist())
            if result.returns is not None
            else None,
            "timestamps": _timestamps(result),
            "trades": [normalize_json_object(trade) for trade in result.trades],
        }
    )
    metadata = normalize_json_object(result.metric_metadata)
    metadata["observations"] = {
        "bar_count": len(observations.equity_curve),
        "trade_count": len(observations.trades),
        "start_date": observations.timestamps[0] if observations.timestamps else None,
        "end_date": observations.timestamps[-1] if observations.timestamps else None,
    }
    scope = PerformanceScope.model_validate(
        {
            "mode": result.mode,
            "values": normalize_json_object(result.metrics),
            "metadata": metadata,
            "split": split,
            "fold_index": fold_index,
            "trial_index": trial_index,
            "parameters": normalize_json_object(parameters or {}),
        }
    )
    return scope, observations


def build_artifacts(
    run_id: str,
    manifest_hash: str | None,
    default_scope: str,
    scopes: dict[str, PerformanceScope],
    observations: dict[str, ScopeObservations],
    *,
    selected_train_scope: str | None = None,
) -> PerformanceArtifacts:
    """Freeze the producer's definitions and complete scope/observation mapping."""
    keys = {key for scope in scopes.values() for key in scope.values}
    definitions = {
        key: MetricDefinitionSnapshot.model_validate(asdict(METRIC_CATALOG[key]))
        for key in sorted(keys)
        if key in METRIC_CATALOG
    }
    return PerformanceArtifacts(
        performance=PerformanceBundle(
            run_id=run_id,
            manifest_hash=manifest_hash,
            default_scope=default_scope,
            scopes=scopes,
            definitions=definitions,
            selected_train_scope=selected_train_scope,
        ),
        observations=ObservationsBundle(
            run_id=run_id, manifest_hash=manifest_hash, scopes=observations
        ),
    )


def validate_projection(
    artifacts: PerformanceArtifacts, projection: dict[str, Any]
) -> None:
    """Reject a numeric result projection that disagrees with saved scope values."""
    bundle = artifacts.performance
    default = bundle.scopes[bundle.default_scope]
    values = default.model_dump(mode="json")["values"]
    if bundle.selected_train_scope is not None:
        train = bundle.scopes[bundle.selected_train_scope]
        train_values = train.model_dump(mode="json")["values"]
        if (
            projection.get("train_metrics") != train_values
            or projection.get("test_metrics") != values
        ):
            raise ValueError(
                "Optimization projection differs from saved train/test metrics"
            )
        if projection.get("best_params") != train.parameters:
            raise ValueError("Optimization projection differs from selected parameters")
        metric = projection.get("optimization_metric")
        if (
            not isinstance(metric, str)
            or metric not in train_values
            or metric not in values
        ):
            raise ValueError("Optimization projection has no valid target metric")
        for key, expected in (
            ("best_train_metric_value", train_values[metric]),
            ("best_oos_metric_value", values[metric]),
        ):
            if projection.get(key) != expected:
                raise ValueError(f"Optimization projection disagrees with {key}")
    elif default.kind == "fold_summary":
        expected = {"n_folds": default.metadata.get("n_folds"), **values}
        if projection != expected:
            raise ValueError("Fold projection differs from saved summary metrics")
    elif projection != values:
        raise ValueError("Result projection differs from saved performance metrics")


def _execution_scope(
    result: BacktestResult,
    config: dict[str, Any],
    *,
    split: Literal["full", "train", "test"],
    fold_index: int | None = None,
    parameters: dict[str, Any] | None = None,
) -> tuple[PerformanceScope, ScopeObservations]:
    scope, observations = scope_from_backtest(
        result, split=split, fold_index=fold_index, parameters=parameters
    )
    metadata = dict(scope.metadata)
    metadata["execution_context"] = normalize_json_object(config.get("backtest", {}))
    return scope.model_copy(update={"metadata": metadata}), observations


def bundles_from_pipeline(
    run_id: str,
    pipeline: PipelineResult,
    projection: dict[str, Any],
) -> PerformanceArtifacts:
    """Create simple or train/test fold scopes from an executed pipeline."""
    scopes: dict[str, PerformanceScope] = {}
    observations: dict[str, ScopeObservations] = {}
    manifest_hash = (
        pipeline.execution_manifest.manifest_hash
        if pipeline.execution_manifest is not None
        else None
    )
    if pipeline.mode == "simple":
        if pipeline.backtest_result is None:
            raise ValueError("Simple pipeline has no backtest observations")
        scopes["full"], observations["full"] = _execution_scope(
            pipeline.backtest_result, pipeline.config_snapshot, split="full"
        )
        if scopes["full"].values != normalize_json_object(projection):
            raise ValueError("Saved metrics differ from executed backtest metrics")
        default_scope = "full"
    elif pipeline.mode == "walk_forward":
        if not pipeline.fold_results:
            raise ValueError("Walk-forward pipeline has no fold observations")
        for fold in pipeline.fold_results:
            splits: tuple[
                tuple[Literal["train", "test"], BacktestResult | None], ...
            ] = (
                ("train", fold.train_backtest),
                ("test", fold.test_backtest),
            )
            for split, result in splits:
                if result is None:
                    raise ValueError("Walk-forward fold has no backtest observations")
                scope_id = f"fold/{fold.fold_index}/{split}"
                if scope_id in scopes:
                    raise ValueError("Duplicate walk-forward fold index")
                scopes[scope_id], observations[scope_id] = _execution_scope(
                    result,
                    pipeline.config_snapshot,
                    split=split,
                    fold_index=fold.fold_index,
                    parameters=fold.strategy_params,
                )
        test_scopes = {
            key: scope for key, scope in scopes.items() if scope.split == "test"
        }
        modes = {scope.mode for scope in test_scopes.values()}
        if len(modes) != 1:
            raise ValueError("Walk-forward folds use different backtest modes")
        metadata: dict[str, JsonValue] = {
            "aggregation": "fold_descriptive",
            "n_folds": len(pipeline.fold_results),
            "fold_metadata": {
                key: scope.metadata for key, scope in test_scopes.items()
            },
        }
        for key in ("settings", "providers", "execution_context"):
            values = [scope.metadata.get(key) for scope in test_scopes.values()]
            if all(value == values[0] for value in values):
                metadata[key] = values[0]
            else:
                metadata[f"{key}_variants"] = values
        if projection.get("n_folds") != len(pipeline.fold_results):
            raise ValueError("Saved fold count differs from executed folds")
        scopes["test_summary"] = PerformanceScope.model_validate(
            {
                "kind": "fold_summary",
                "mode": modes.pop(),
                "values": {
                    key: value
                    for key, value in normalize_json_object(projection).items()
                    if key != "n_folds"
                },
                "metadata": metadata,
                "split": "test",
            }
        )
        default_scope = "test_summary"
    else:
        raise ValueError(f"Unsupported pipeline mode: {pipeline.mode}")
    return build_artifacts(run_id, manifest_hash, default_scope, scopes, observations)
