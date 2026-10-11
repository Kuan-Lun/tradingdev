"""Validate execution inputs or normalize research outputs without losing keys."""

from __future__ import annotations

import math
from datetime import date
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Mapping

    from pydantic import JsonValue


def strict_json_value(value: object, *, allow_dates: bool = False) -> JsonValue:
    """Copy finite JSON, rejecting lossy values and recursive containers.

    Execution inputs must fail instead of replacing non-finite numbers with null
    or converting object keys to strings. YAML dates may be accepted explicitly;
    their copies use ISO strings. Shared container aliases are valid, but cycles
    and unsupported Python objects are not. The original input is never mutated.
    """
    active: set[int] = set()

    def copy(node: object) -> JsonValue:
        if node is None or isinstance(node, bool | str | int):
            return node
        if isinstance(node, float):
            if not math.isfinite(node):
                raise ValueError("JSON values must contain only finite numbers")
            return node
        if allow_dates and isinstance(node, date):
            return node.isoformat()
        if not isinstance(node, list | dict):
            raise ValueError(f"Unsupported JSON value: {type(node).__name__}")
        identity = id(node)
        if identity in active:
            raise ValueError("JSON values must not contain circular references")
        active.add(identity)
        try:
            if isinstance(node, list):
                return [copy(item) for item in node]
            result: dict[str, JsonValue] = {}
            for key, item in node.items():
                if not isinstance(key, str):
                    raise ValueError("JSON object keys must be strings")
                result[key] = copy(item)
            return result
        finally:
            active.remove(identity)

    try:
        return copy(value)
    except RecursionError as exc:
        raise ValueError("JSON value nesting is too deep") from exc


def normalize_json_value(value: object) -> JsonValue:
    """Preserve JSON values, converting non-finite numbers to null recursively.

    NumPy boolean, integer, and floating scalars become their Python equivalents.
    Tuples become JSON arrays, matching the standard encoder. Unknown objects and
    non-string object keys raise TypeError instead of losing data through string
    conversion. Every container in the returned value is newly allocated.
    """
    if value is None:
        return None
    if isinstance(value, bool | np.bool_):
        return bool(value)
    if isinstance(value, str):
        return str(value)
    if isinstance(value, int | np.integer):
        return int(value)
    if isinstance(value, float | np.floating):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, dict):
        return normalize_json_object(value)
    if isinstance(value, list | tuple):
        return [normalize_json_value(item) for item in value]
    msg = f"Unsupported JSON value type: {type(value).__name__}"
    raise TypeError(msg)


def normalize_json_object(value: Mapping[str, object]) -> dict[str, JsonValue]:
    """Normalize a JSON object's values and require lossless string keys."""
    normalized: dict[str, JsonValue] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            msg = f"JSON object keys must be strings, got {type(key).__name__}"
            raise TypeError(msg)
        normalized[key] = normalize_json_value(item)
    return normalized
