"""Shared error responses for strategy lifecycle and read operations."""

from __future__ import annotations

from typing import Any

from tradingdev.adapters.storage.strategy_revisions import (
    StrategyRevisionError,
    UnsupportedStrategyRevisionError,
)


def strategy_not_found(strategy_id: str, revision_id: str | None) -> dict[str, Any]:
    return {
        "success": False,
        "error": f"Strategy not found: {strategy_id}, revision={revision_id}",
        "code": "strategy_not_found",
    }


def strategy_revision_error(exc: StrategyRevisionError) -> dict[str, Any]:
    return {
        "success": False,
        "error": str(exc),
        "code": (
            "strategy_revision_required"
            if isinstance(exc, UnsupportedStrategyRevisionError)
            else "strategy_revision_invalid"
        ),
    }
