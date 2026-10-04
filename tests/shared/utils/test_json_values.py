"""Strict execution JSON validation preserves input identity and rejects loss."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

import pytest

from tradingdev.shared.utils.json_values import strict_json_value


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ({"fees": float("nan")}, "finite"),
        ({"nested": [float("inf")]}, "finite"),
        ({"nested": [-float("inf")]}, "finite"),
        ({"nested": {1: "integer", "1": "string"}}, "keys must be strings"),
        ({"nested": object()}, "Unsupported JSON value"),
        ({"nested": (1, 2)}, "Unsupported JSON value"),
        ({"nested": {1, 2}}, "Unsupported JSON value"),
    ],
)
def test_strict_json_rejects_values_that_serializers_might_coerce(
    value: object, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        strict_json_value(value)


@pytest.mark.parametrize("container", ["list", "dict"])
def test_cycles_fail_with_a_validation_error(container: str) -> None:
    recursive: Any = [] if container == "list" else {}
    if container == "list":
        recursive.append(recursive)
    else:
        recursive["self"] = recursive
    with pytest.raises(ValueError, match="circular references"):
        strict_json_value(recursive)


def test_shared_aliases_are_copied_without_rejecting_a_noncyclic_graph() -> None:
    shared = [1, {"flag": True}]
    source = {"first": shared, "second": shared}
    copied = strict_json_value(source)
    assert copied == source
    assert isinstance(copied, dict)
    assert copied["first"] is not shared
    assert copied["first"] is not copied["second"]
    assert source["first"] is source["second"]


@pytest.mark.parametrize("value", [date(2024, 1, 1), datetime(2024, 1, 1)])
def test_yaml_dates_require_an_explicit_policy(value: date) -> None:
    source = {"start_date": value}
    with pytest.raises(ValueError, match="Unsupported JSON value"):
        strict_json_value(source)
    assert strict_json_value(source, allow_dates=True) == {
        "start_date": value.isoformat()
    }
    assert source["start_date"] is value
