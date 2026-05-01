"""Helpers for extracting run lineage metadata from YAML configs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config_payload(config_path: Path) -> dict[str, Any] | None:
    """Load a YAML config as a mapping, returning None on invalid input."""
    if not config_path.exists():
        return None
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None
    return raw if isinstance(raw, dict) else None


def resolve_strategy_source(
    raw: dict[str, Any] | None,
) -> Path | None:
    """Resolve ``strategy.source_path`` relative to the current project."""
    if raw is None:
        return None
    strategy = raw.get("strategy", {})
    if not isinstance(strategy, dict):
        return None
    source = strategy.get("source_path")
    if not source:
        return None
    path = Path(str(source))
    if not path.is_absolute():
        path = Path.cwd() / path
    return path.resolve()


def extract_random_seed(raw: dict[str, Any] | None) -> int | None:
    """Return the explicit run seed, or a unique model seed from config."""
    if raw is None:
        return None
    for candidate in (
        raw.get("random_seed"),
        _mapping_value(raw.get("backtest"), "random_seed"),
    ):
        parsed = _parse_seed(candidate)
        if parsed is not None:
            return parsed

    strategy = raw.get("strategy")
    params = _mapping_value(strategy, "parameters")
    seeds = _collect_seed_values(params)
    return seeds[0] if len(seeds) == 1 else None


def _mapping_value(value: object, key: str) -> object | None:
    if isinstance(value, dict):
        return value.get(key)
    return None


def _collect_seed_values(value: object) -> list[int]:
    found: set[int] = set()

    def walk(node: object) -> None:
        if isinstance(node, dict):
            for key, item in node.items():
                if key in {"random_seed", "random_state", "seed"}:
                    parsed = _parse_seed(item)
                    if parsed is not None:
                        found.add(parsed)
                walk(item)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(value)
    return sorted(found)


def _parse_seed(value: object) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None
