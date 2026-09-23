"""Helpers for extracting run lineage metadata from YAML configs."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from tradingdev.app.job_config import bind_strategy_revision
from tradingdev.app.strategy_service import StrategyNotExecutableError, StrategyService

if TYPE_CHECKING:
    from tradingdev.adapters.storage.filesystem import WorkspacePaths


@dataclass(frozen=True)
class StrategySourceSnapshot:
    """One verified read of the exact source retained in run artifacts."""

    revision_id: str | None
    path: Path | None
    content: bytes | None
    source_hash: str | None


def read_strategy_snapshot(
    raw: dict[str, Any] | None,
    workspace: WorkspacePaths,
    *,
    strategy_id: str,
) -> StrategySourceSnapshot:
    """Read the pinned revision, checking its identity and original digest."""
    strategy = raw.get("strategy", {}) if raw is not None else {}
    revision_id = strategy.get("revision_id") if isinstance(strategy, dict) else None
    expected_hash = strategy.get("source_hash") if isinstance(strategy, dict) else None
    if expected_hash is not None and not isinstance(expected_hash, str):
        msg = "Invalid strategy source hash in result configuration"
        raise StrategyNotExecutableError(msg)
    if revision_id is not None:
        if not isinstance(revision_id, str) or raw is None:
            msg = "Invalid strategy revision in result configuration"
            raise StrategyNotExecutableError(msg)
        spec = StrategyService(workspace).load(strategy_id, revision_id)
        if spec is None:
            msg = "Strategy revision not found while saving run artifacts"
            raise StrategyNotExecutableError(msg)
        bind_strategy_revision(raw, spec)
        expected_hash = str(strategy["source_hash"])
    source = resolve_strategy_source(raw)
    content = source.read_bytes() if source is not None and source.is_file() else None
    digest = hashlib.sha256(content).hexdigest() if content is not None else None
    if expected_hash is not None and digest != expected_hash:
        msg = "Strategy source hash changed before artifact persistence"
        raise StrategyNotExecutableError(msg)
    return StrategySourceSnapshot(revision_id, source, content, digest)


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
