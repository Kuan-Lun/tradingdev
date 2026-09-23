"""Normalize research values into finite, standard JSON without mutating inputs."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Mapping

    from pydantic import JsonValue


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
