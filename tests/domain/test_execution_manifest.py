"""Integrity and canonicalization of complete execution requests."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from copy import deepcopy
from typing import Any

import pytest
from pydantic import ValidationError

from tradingdev.domain.execution import (
    ExecutionManifest,
    ManifestError,
    OptimizationSpec,
)


def _config() -> dict[str, Any]:
    return {
        "strategy": {
            "id": "example",
            "revision_id": "revision_a",
            "parameters": {"windows": [2, 10]},
        },
        "backtest": {
            "symbol": "BTCUSDT",
            "timeframe": "1h",
            "start_date": "2024-01-01",
            "end_date": "2024-04-30",
            "init_cash": 10_000,
        },
        "data": {"raw_dir": "/resolved/raw", "processed_dir": "/resolved/processed"},
    }


def _search(**overrides: Any) -> OptimizationSpec:
    values: dict[str, Any] = {
        "param_ranges": {"slow": [20, 10], "fast": [2, 4]},
        "optimization_metric": "sharpe_ratio",
        "train_start": "2024-01-01",
        "train_end": "2024-02-29",
        "test_start": "2024-03-01",
        "test_end": "2024-04-30",
    }
    values.update(overrides)
    return OptimizationSpec.model_validate(values)


def test_manifest_roundtrip_and_independent_canonical_digest() -> None:
    manifest = ExecutionManifest.create(kind="backtest", config=_config())
    payload = manifest.model_dump(mode="json", exclude={"manifest_hash"})
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()

    assert manifest.manifest_hash == hashlib.sha256(canonical).hexdigest()
    assert ExecutionManifest.model_validate_json(manifest.model_dump_json()) == manifest
    manifest.verify(expected_hash=manifest.manifest_hash)
    assert "job_id" not in payload
    assert "created_at" not in payload


def test_implicit_defaults_match_explicit_defaults_without_changing_input() -> None:
    original = _config()
    before = deepcopy(original)
    implicit = ExecutionManifest.create(kind="backtest", config=original)
    explicit = ExecutionManifest.create(kind="backtest", config=implicit.config_copy())
    nullable = _config()
    nullable.update(validation=None, random_seed=None, parallel=None)
    with_nulls = ExecutionManifest.create(kind="backtest", config=nullable)

    assert implicit.manifest_hash == explicit.manifest_hash == with_nulls.manifest_hash
    assert original == before
    assert implicit.config["strategy"] == original["strategy"]
    assert implicit.config["validation"] is None
    assert implicit.config["random_seed"] is None
    assert implicit.config["parallel"] == {
        "reserve_cores": 2,
        "safety_factor": 0.6,
        "overhead_multiplier": 3.0,
    }
    backtest = implicit.config["backtest"]
    assert isinstance(backtest, dict)
    assert backtest["fees"] == 0.0006
    assert backtest["slippage"] == 0.0005
    assert backtest["mode"] == "signal"


def test_dates_and_dictionary_order_have_stable_hashes() -> None:
    original = _config()
    equivalent = dict(reversed(list(original.items())))
    equivalent["backtest"] = dict(reversed(list(original["backtest"].items())))
    equivalent["backtest"]["start_date"] = dt.date(2024, 1, 1)
    equivalent["backtest"]["end_date"] = dt.datetime(2024, 4, 30)
    first = ExecutionManifest.create(kind="backtest", config=original)
    second = ExecutionManifest.create(kind="backtest", config=equivalent)
    assert first.manifest_hash == second.manifest_hash


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("strategy", "revision_id", "revision_b"),
        ("strategy", "parameters", {"windows": [3, 10]}),
        ("backtest", "fees", 0.001),
        ("backtest", "slippage", 0.002),
        ("backtest", "mode", "volume"),
        ("backtest", "random_seed", 12),
        ("backtest", "end_date", "2024-04-29"),
        ("data", "raw_dir", "/another/raw"),
    ],
)
def test_execution_choices_change_digest(section: str, field: str, value: Any) -> None:
    original = _config()
    changed = deepcopy(original)
    changed[section][field] = value
    assert ExecutionManifest.create(kind="backtest", config=original).manifest_hash != (
        ExecutionManifest.create(kind="backtest", config=changed).manifest_hash
    )


def test_top_level_seed_and_parallel_policy_change_digest() -> None:
    config = _config()
    original = ExecutionManifest.create(kind="backtest", config=config)
    config["random_seed"] = 42
    seeded = ExecutionManifest.create(kind="backtest", config=config)
    config["parallel"] = {"reserve_cores": 1}
    parallel = ExecutionManifest.create(kind="backtest", config=config)
    assert (
        len({original.manifest_hash, seeded.manifest_hash, parallel.manifest_hash}) == 3
    )


def test_manifest_owns_inputs_and_returns_independent_nested_config() -> None:
    config = _config()
    search = _search()
    manifest = ExecutionManifest.create(
        kind="optimization",
        config=config,
        optimization=search,
    )
    config["strategy"]["parameters"]["windows"].append(100)
    search.param_ranges["fast"].append(99)
    copied = manifest.config_copy()
    copied["strategy"]["parameters"]["windows"].append(200)

    manifest.verify()
    assert manifest.config_copy()["strategy"]["parameters"]["windows"] == [2, 10]
    assert manifest.optimization is not None
    assert manifest.optimization.param_ranges["fast"] == [2, 4]


def test_nested_mutation_and_wrong_expected_hash_are_rejected() -> None:
    manifest = ExecutionManifest.create(kind="backtest", config=_config())
    with pytest.raises(ManifestError, match="expected hash"):
        manifest.verify("0" * 64)
    strategy = manifest.config["strategy"]
    assert isinstance(strategy, dict)
    strategy["revision_id"] = "modified"
    with pytest.raises(ManifestError, match="hash does not match"):
        manifest.verify()
    with pytest.raises(ManifestError, match="hash does not match"):
        manifest.config_copy()


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_config_and_ranges_are_rejected(value: float) -> None:
    config = _config()
    config["strategy"]["parameters"]["nested"] = [value]
    with pytest.raises(ManifestError, match="finite"):
        ExecutionManifest.create(kind="backtest", config=config)
    with pytest.raises(ValidationError, match="finite"):
        _search(param_ranges={"fast": [{"nested": value}]})


def test_nonfinite_search_mutation_is_not_serialized_to_null() -> None:
    manifest = ExecutionManifest.create(
        kind="optimization",
        config=_config(),
        optimization=_search(param_ranges={"optional": [None]}),
    )
    assert manifest.optimization is not None
    manifest.optimization.param_ranges["optional"][0] = float("nan")
    with pytest.raises(ManifestError, match="finite"):
        manifest.verify()


@pytest.mark.parametrize("invalid", [{1: "bad"}, {"bad": object()}, {"bad": (1, 2)}])
def test_non_json_values_are_rejected_without_lossy_coercion(invalid: Any) -> None:
    config = _config()
    config["extra"] = invalid
    with pytest.raises(ManifestError):
        ExecutionManifest.create(kind="backtest", config=config)


def test_unknown_version_extra_fields_and_tampered_content_cannot_load() -> None:
    manifest = ExecutionManifest.create(kind="backtest", config=_config())
    payload = manifest.model_dump(mode="json")
    for changes in (
        {"schema_version": 2},
        {"unexpected": True},
        {"kind": "optimization"},
    ):
        with pytest.raises(ValidationError):
            ExecutionManifest.model_validate_json(json.dumps(payload | changes))
    payload["config"]["backtest"]["fees"] = 1.0
    with pytest.raises(ValidationError, match="hash does not match"):
        ExecutionManifest.model_validate_json(json.dumps(payload))


def test_modes_require_matching_validation_and_optimization() -> None:
    config = _config()
    with pytest.raises(ManifestError, match="validation"):
        ExecutionManifest.create(kind="walk_forward", config=config)
    with pytest.raises(ManifestError, match="optimization spec"):
        ExecutionManifest.create(kind="optimization", config=config)
    with pytest.raises(ManifestError, match="optimization spec"):
        ExecutionManifest.create(kind="backtest", config=config, optimization=_search())
    config["validation"] = {"n_splits": 3}
    walk_forward = ExecutionManifest.create(kind="walk_forward", config=config)
    assert walk_forward.config_copy()["validation"]["train_ratio"] == 0.8
    for kind in ("backtest", "optimization"):
        with pytest.raises(ManifestError, match="validation"):
            ExecutionManifest.create(kind=kind, config=config)


def test_search_order_defaults_roundtrip_and_derived_combination_count() -> None:
    search = _search()
    assert list(search.param_ranges) == ["fast", "slow"]
    assert search.param_ranges["slow"] == [20, 10]
    assert search.total_combinations == 4
    assert "total_combinations" not in search.model_dump()
    manifest = ExecutionManifest.create(
        kind="optimization",
        config=_config(),
        optimization=search,
    )
    assert ExecutionManifest.model_validate_json(manifest.model_dump_json()) == manifest
    assert search.direction == "maximize"
    assert search.trial_timeout_seconds == 300
    assert search.confirmation_timeout_seconds == 1800
    assert search.confirmation_poll_interval == 2.0


@pytest.mark.parametrize(
    "change",
    [
        {"param_ranges": {"slow": [10, 20], "fast": [2, 4]}},
        {"optimization_metric": "total_return"},
        {"train_end": "2024-02-28"},
        {"test_start": "2024-03-02"},
        {"trial_timeout_seconds": 301},
        {"confirmation_timeout_seconds": 1801},
        {"confirmation_poll_interval": 1.0},
    ],
)
def test_search_and_confirmation_choices_change_digest(change: dict[str, Any]) -> None:
    original = ExecutionManifest.create(
        kind="optimization",
        config=_config(),
        optimization=_search(),
    )
    changed = ExecutionManifest.create(
        kind="optimization",
        config=_config(),
        optimization=_search(**change),
    )
    assert original.manifest_hash != changed.manifest_hash


@pytest.mark.parametrize(
    "change",
    [
        {"param_ranges": {}},
        {"param_ranges": {"fast": []}},
        {"optimization_metric": "unknown"},
        {"direction": "minimize"},
        {"trial_timeout_seconds": 0},
        {"trial_timeout_seconds": True},
        {"confirmation_timeout_seconds": -1},
        {"confirmation_poll_interval": float("inf")},
        {"confirmation_poll_interval": 0},
        {"train_end": "2024-03-01"},
        {"test_end": "2024-03-01"},
        {"test_start": "2024-03-01T00:00:00"},
        {"train_start": "20240101"},
        {"unexpected": True},
    ],
)
def test_invalid_search_requests_are_rejected(change: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        _search(**change)
