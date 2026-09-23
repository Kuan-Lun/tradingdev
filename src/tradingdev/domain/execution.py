"""Complete, hashed execution requests independent of jobs and storage."""

from __future__ import annotations

import datetime as dt  # noqa: TC003
import hashlib
import json
import math
from copy import deepcopy
from typing import Any, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    ValidationError,
    field_validator,
    model_validator,
)

from tradingdev.domain.backtest.schemas import BacktestRunConfig, ParallelConfig

type ExecutionKind = Literal["backtest", "walk_forward", "optimization"]
type OptimizationMetric = Literal[
    "total_return",
    "total_pnl",
    "annual_return",
    "sharpe_ratio",
    "max_drawdown",
    "win_rate",
    "profit_factor",
]


class ManifestError(ValueError):
    """An execution request is invalid or differs from its recorded digest."""


def _json_value(value: object, *, allow_dates: bool = False) -> JsonValue:
    """Copy finite JSON without coercing keys or silently replacing numbers."""
    if value is None or isinstance(value, bool | str | int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ManifestError("Execution values must contain only finite numbers")
        return value
    if allow_dates and isinstance(value, dt.date):
        return value.isoformat()
    if isinstance(value, list):
        return [_json_value(item, allow_dates=allow_dates) for item in value]
    if isinstance(value, dict):
        copied: dict[str, JsonValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ManifestError("Execution JSON object keys must be strings")
            copied[key] = _json_value(item, allow_dates=allow_dates)
        return copied
    raise ManifestError(f"Unsupported execution value: {type(value).__name__}")


def _resolve_config(value: object) -> dict[str, JsonValue]:
    copied = _json_value(value, allow_dates=True)
    if not isinstance(copied, dict):
        raise ManifestError("Execution config must be an object")
    try:
        config = BacktestRunConfig.model_validate(copied)
        parallel_value = copied.get("parallel")
        parallel = ParallelConfig.model_validate(
            {} if parallel_value is None else parallel_value
        )
    except ValidationError as exc:
        raise ManifestError(f"Invalid execution config: {exc}") from exc
    resolved = config.model_dump(mode="python")
    resolved["parallel"] = parallel.model_dump(mode="python")
    result = _json_value(resolved, allow_dates=True)
    assert isinstance(result, dict)
    return result


class OptimizationSpec(BaseModel):
    """The complete search and confirmation policy for one optimization."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    param_ranges: dict[str, list[JsonValue]]
    optimization_metric: OptimizationMetric
    train_start: dt.date
    train_end: dt.date
    test_start: dt.date
    test_end: dt.date
    direction: Literal["maximize"] = "maximize"
    trial_timeout_seconds: int = Field(default=300, gt=0, strict=True)
    confirmation_timeout_seconds: int = Field(default=1800, gt=0, strict=True)
    confirmation_poll_interval: float = Field(default=2.0, gt=0, strict=True)

    @field_validator("param_ranges", mode="before")
    @classmethod
    def finite_nonempty_ranges(cls, value: object) -> dict[str, list[JsonValue]]:
        """Fix parameter traversal order while preserving each candidate order."""
        copied = _json_value(value)
        if not isinstance(copied, dict) or not copied:
            raise ManifestError("param_ranges must be a non-empty object")
        ranges: dict[str, list[JsonValue]] = {}
        for name in sorted(copied):
            values = copied[name]
            if not isinstance(values, list) or not values:
                raise ManifestError(f"param_ranges['{name}'] must be a non-empty list")
            ranges[name] = values
        return ranges

    @field_validator(
        "train_start", "train_end", "test_start", "test_end", mode="before"
    )
    @classmethod
    def calendar_day(cls, value: object) -> dt.date:
        """Reject timestamps, since search boundaries are inclusive dates."""
        if isinstance(value, dt.date) and not isinstance(value, dt.datetime):
            return value
        if isinstance(value, str):
            try:
                parsed = dt.date.fromisoformat(value)
            except ValueError as exc:
                raise ManifestError("Optimization dates must use YYYY-MM-DD") from exc
            if parsed.isoformat() == value:
                return parsed
        raise ManifestError("Optimization dates must use YYYY-MM-DD")

    @model_validator(mode="after")
    def ordered_periods(self) -> Self:
        """Keep training and test calendar days ordered and disjoint."""
        if not (self.train_start < self.train_end < self.test_start < self.test_end):
            raise ManifestError(
                "Dates must satisfy: train_start < train_end < test_start < test_end "
                "(inclusive calendar days, without overlap)"
            )
        return self

    @property
    def total_combinations(self) -> int:
        """Derive search size without persisting duplicate state."""
        return math.prod(len(values) for values in self.param_ranges.values())


class ExecutionManifest(BaseModel):
    """A versioned execution request with a digest over all effective settings.

    Pydantic freezes field assignment, not nested dictionaries and lists. Consumers
    must verify before use and mutate only the independent ``config_copy()`` value.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    schema_version: Literal[1] = 1
    kind: ExecutionKind
    config: dict[str, JsonValue]
    optimization: OptimizationSpec | None = None
    manifest_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("config", mode="before")
    @classmethod
    def resolved_config(cls, value: object) -> dict[str, JsonValue]:
        """Normalize dates and fill execution defaults on creation and loading."""
        return _resolve_config(value)

    @model_validator(mode="after")
    def valid_manifest(self) -> Self:
        """A decoded manifest must satisfy its mode and recorded digest."""
        self.verify()
        return self

    @classmethod
    def create(
        cls,
        *,
        kind: ExecutionKind,
        config: dict[str, Any],
        optimization: OptimizationSpec | None = None,
    ) -> Self:
        """Resolve an independent request and compute its canonical SHA-256."""
        try:
            resolved = _resolve_config(config)
            search = (
                OptimizationSpec.model_validate(optimization.model_dump(mode="python"))
                if optimization is not None
                else None
            )
            payload = {
                "schema_version": 1,
                "kind": kind,
                "config": resolved,
                "optimization": search.model_dump(mode="python") if search else None,
            }
            return cls(
                kind=kind,
                config=resolved,
                optimization=search,
                manifest_hash=cls._digest(payload),
            )
        except (TypeError, ValueError) as exc:
            if isinstance(exc, ManifestError):
                raise
            raise ManifestError(f"Invalid execution manifest: {exc}") from exc

    @staticmethod
    def _digest(payload: dict[str, Any]) -> str:
        finite_payload = _json_value(payload, allow_dates=True)
        encoded = json.dumps(
            finite_payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def verify(self, expected_hash: str | None = None) -> None:
        """Reject invalid mode, nested mutations, and a mismatching job digest."""
        is_walk_forward = self.config.get("validation") is not None
        if is_walk_forward != (self.kind == "walk_forward"):
            raise ManifestError("Execution kind does not match config.validation")
        if (self.optimization is not None) != (self.kind == "optimization"):
            raise ManifestError(
                "Only optimization execution requires an optimization spec"
            )
        if expected_hash is not None and expected_hash != self.manifest_hash:
            raise ManifestError("Execution manifest does not match the expected hash")
        payload = self.model_dump(mode="python", exclude={"manifest_hash"})
        if self._digest(payload) != self.manifest_hash:
            raise ManifestError("Execution manifest hash does not match its content")

    def config_copy(self) -> dict[str, Any]:
        """Return an independent execution config after checking its integrity."""
        self.verify()
        return deepcopy(self.config)
