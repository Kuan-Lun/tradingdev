"""Materialize the exact configuration used by a background job."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    from pathlib import Path


def apply_run_overrides(
    raw_config: dict[str, Any],
    *,
    symbol: str,
    timeframe: str,
    start_date: str,
    end_date: str,
) -> dict[str, Any]:
    """Apply request-level market and date choices without editing the strategy."""
    effective_config = deepcopy(raw_config)
    backtest = effective_config.get("backtest")
    if not isinstance(backtest, dict):
        msg = "backtest config must be a mapping"
        raise ValueError(msg)
    backtest.update(
        symbol=symbol,
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date,
    )
    data = effective_config.get("data")
    if isinstance(data, dict):
        requirements = data.get("requirements")
        if isinstance(requirements, dict):
            market = requirements.get("market")
            if isinstance(market, dict):
                market.update(symbol=symbol, timeframe=timeframe)
    return effective_config


def write_job_config(run_dir: Path, config: dict[str, Any]) -> Path:
    """Save the execution snapshot in the job's own artifact directory."""
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "config.yaml"
    path.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    return path
