"""Bounded preparation requests and explicit sample-execution evidence."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

# Pydantic resolves both annotations when validating the wire schema.
from tradingdev.domain.execution import ExecutionKind, ExecutionManifest  # noqa: TC001

type PreflightCheck = Literal[
    "configuration",
    "signal_contract",
    "signals",
    "engine",
    "serialization",
    "fit",
    "candidate_binding",
]


class PreflightModel(BaseModel):
    """Reject ambiguous values in preparation requests and saved evidence."""

    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class PreflightRequest(PreflightModel):
    """Prepare service arguments with a total sample budget and declared warm-up."""

    kind: ExecutionKind
    arguments: dict[str, JsonValue]
    minimum_history_bars: int = Field(ge=1, le=4096)
    sample_bars: int = Field(default=1024, ge=64, le=4096)

    @model_validator(mode="after")
    def sufficient_budget(self) -> Self:
        if self.minimum_history_bars > self.sample_bars:
            raise ValueError("minimum_history_bars exceeds the total sample budget")
        return self


class PreflightWindow(PreflightModel):
    """The actual data passed to one sampled execution path."""

    role: Literal["full", "train", "test"]
    start: str
    end: str
    rows: int = Field(ge=1)


class PreflightReceipt(PreflightModel):
    """Successful sample evidence, never a promise about the complete run."""

    manifest_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    elapsed_seconds: float = Field(ge=0)
    sample_bars_requested: int = Field(ge=64, le=4096)
    sample_bars_used: int = Field(ge=1, le=4096)
    minimum_history_bars: int = Field(ge=1, le=4096)
    data_origin: Literal["market_data"] = "market_data"
    data_source: str
    windows: list[PreflightWindow]
    checked_paths: list[PreflightCheck]
    trade_count: int = Field(ge=0)
    execution_record_count: int | None = Field(default=None, ge=0)
    trading_path_exercised: bool
    tested_fold_count: int | None = Field(default=None, ge=1)
    total_fold_count: int | None = Field(default=None, ge=1)
    tested_candidates: int | None = Field(default=None, ge=1)
    total_candidates: int | None = Field(default=None, ge=1)
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def consistent_coverage(self) -> Self:
        if (
            not self.windows
            or sum(window.rows for window in self.windows) != self.sample_bars_used
        ):
            raise ValueError("Preflight window rows differ from the sample count")
        if self.sample_bars_used > self.sample_bars_requested:
            raise ValueError("Preflight exceeded its total sample budget")
        if any(window.rows < self.minimum_history_bars for window in self.windows):
            raise ValueError("Preflight window is shorter than the declared warm-up")
        if self.trading_path_exercised != (self.trade_count > 0):
            raise ValueError("Trading coverage must agree with observed trades")
        for tested, total in (
            (self.tested_fold_count, self.total_fold_count),
            (self.tested_candidates, self.total_candidates),
        ):
            if (tested is None) != (total is None) or (
                tested is not None and total is not None and tested > total
            ):
                raise ValueError("Invalid preflight execution coverage")
        return self


class PreflightPayload(PreflightModel):
    """Verified child-process output independent of job and plan storage."""

    manifest: ExecutionManifest
    original_config_path: str
    receipt: PreflightReceipt

    @model_validator(mode="after")
    def matching_manifest(self) -> Self:
        self.manifest.verify(expected_hash=self.receipt.manifest_hash)
        return self
