"""Explicit constructor settings captured before a strategy is executed."""

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, JsonValue, field_validator


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

    @field_validator("constructor_kwargs", mode="before")
    @classmethod
    def finite_constructor_kwargs(cls, value: object) -> dict[str, JsonValue]:
        """Reject lossy serialization and detach the caller's dictionaries."""
        copied = finite_strategy_value(value)
        if not isinstance(copied, dict):
            raise ValueError("Strategy constructor settings must be an object")
        return copied
