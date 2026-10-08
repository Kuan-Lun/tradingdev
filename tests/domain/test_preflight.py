"""Preflight contracts reject missing warm-up and contradictory coverage."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from tradingdev.domain.preflight import PreflightReceipt, PreflightRequest


@pytest.mark.parametrize(
    "fields",
    [
        {},
        {"minimum_history_bars": True},
        {"minimum_history_bars": 0},
        {"minimum_history_bars": 1025},
        {"minimum_history_bars": 1, "sample_bars": 4097},
    ],
)
def test_request_requires_explicit_feasible_history(fields: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        PreflightRequest.model_validate({"kind": "backtest", "arguments": {}, **fields})


@pytest.mark.parametrize(
    "changed",
    [
        {"sample_bars_used": 63},
        {"minimum_history_bars": 65},
        {"trading_path_exercised": True},
        {"tested_fold_count": 1},
        {"tested_candidates": 3, "total_candidates": 2},
        {"checked_paths": ["invented_check"]},
        {"data_origin": "synthetic"},
    ],
)
def test_receipt_rejects_contradictory_evidence(changed: dict[str, Any]) -> None:
    receipt: dict[str, Any] = {
        "manifest_hash": "a" * 64,
        "elapsed_seconds": 1.0,
        "sample_bars_requested": 64,
        "sample_bars_used": 64,
        "minimum_history_bars": 8,
        "data_source": "binance_vision",
        "windows": [
            {"role": "full", "start": "2024-01-01", "end": "2024-01-03", "rows": 64}
        ],
        "checked_paths": ["configuration", "signals", "engine", "serialization"],
        "trade_count": 0,
        "trading_path_exercised": False,
    }
    with pytest.raises(ValidationError):
        PreflightReceipt.model_validate({**receipt, **changed})
