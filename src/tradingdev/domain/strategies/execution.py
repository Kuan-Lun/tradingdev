"""Explicit constructor settings captured before a strategy is executed."""

from __future__ import annotations

import math
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    field_validator,
    model_validator,
)

from tradingdev.domain.runtime import RuntimeSpec


def finite_strategy_value(value: object) -> JsonValue:
    """Copy settings without coercing unsupported defaults into JSON values."""
    if value is None or isinstance(value, str | bool | int):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    if isinstance(value, list):
        return [finite_strategy_value(item) for item in value]
    if isinstance(value, dict):
        copied: dict[str, JsonValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("Strategy execution object keys must be strings")
            copied[key] = finite_strategy_value(item)
        return copied
    raise ValueError(
        "Strategy execution settings require finite JSON values; "
        f"received {type(value).__name__}"
    )


def merge_strategy_parameters(
    captured: dict[str, JsonValue], overrides: object
) -> dict[str, JsonValue]:
    """Return detached effective values using the execution overlay semantics.

    Objects recursively replace only named existing entries, so an empty object
    preserves a captured object. Other JSON values replace the whole entry,
    including arrays. Neither argument is modified.
    """
    target = finite_strategy_value(captured)
    updates = finite_strategy_value(overrides)
    if not isinstance(target, dict) or not isinstance(updates, dict):
        raise ValueError("Strategy parameters must be a mapping")
    _overlay_parameters(target, updates)
    return target


def _overlay_parameters(
    target: dict[str, JsonValue], overrides: dict[str, JsonValue]
) -> None:
    for name, value in overrides.items():
        if name not in target:
            raise ValueError(f"Unknown strategy parameter: {name}")
        previous = target[name]
        if isinstance(previous, dict) and isinstance(value, dict):
            _overlay_parameters(previous, value)
        else:
            target[name] = value


class GeneratedProgram(BaseModel):
    """Bind generated constructor settings to source and an immutable runtime."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    class_name: str = Field(min_length=1, max_length=256)
    runtime: RuntimeSpec


class StrategyExecution(BaseModel):
    """Effective values, excluding the engine and parallel policy injections.

    Bundled strategies store their fully expanded ``config`` and optional
    ``fit_config`` objects. Generated strategies store named constructor arguments
    with every declared default made explicit. The enclosing execution manifest
    protects these mutable nested values with its digest.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    kind: Literal["bundled", "generated"]
    constructor_kwargs: dict[str, JsonValue]
    program: GeneratedProgram | None = Field(
        default=None, exclude_if=lambda value: value is None
    )

    @model_validator(mode="after")
    def require_program_identity(self) -> StrategyExecution:
        if (self.kind == "generated") != (self.program is not None):
            raise ValueError(
                "Generated executions require a fixed source and isolated runtime; "
                "bundled executions cannot include a generated program"
            )
        return self

    @field_validator("constructor_kwargs", mode="before")
    @classmethod
    def finite_constructor_kwargs(cls, value: object) -> dict[str, JsonValue]:
        """Reject lossy serialization and detach the caller's dictionaries."""
        copied = finite_strategy_value(value)
        if not isinstance(copied, dict):
            raise ValueError("Strategy constructor settings must be an object")
        return copied
