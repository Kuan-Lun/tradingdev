"""Utilities for parallel grid-search execution."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import psutil

if TYPE_CHECKING:
    import pandas as pd

from btc_strategy.utils.logger import setup_logger

logger = setup_logger(__name__)


def estimate_n_jobs(
    df: pd.DataFrame,
    safety_factor: float = 0.6,
    overhead_multiplier: float = 3.0,
) -> int:
    """Estimate safe number of parallel workers based on available memory.

    Each worker receives a copy of *df* plus intermediate allocations
    (signals DataFrame, equity curve, etc.).  We conservatively assume
    each worker needs ``overhead_multiplier`` × the raw DataFrame size.

    Args:
        df: The DataFrame that will be distributed to workers.
        safety_factor: Fraction of available RAM to use (0–1).
        overhead_multiplier: Estimated memory multiplier per worker.

    Returns:
        Number of workers (>= 1, <= cpu_count).
    """
    per_worker_bytes = int(df.memory_usage(deep=True).sum() * overhead_multiplier)
    available_bytes = int(psutil.virtual_memory().available * safety_factor)
    max_by_memory = max(1, available_bytes // max(per_worker_bytes, 1))
    max_by_cpu = os.cpu_count() or 4

    n_jobs = min(max_by_memory, max_by_cpu)

    logger.info(
        "Parallel estimate: df=%.1f MB, available=%.1f MB, "
        "n_jobs=%d (mem_limit=%d, cpu=%d)",
        per_worker_bytes / overhead_multiplier / 1e6,
        available_bytes / 1e6,
        n_jobs,
        max_by_memory,
        max_by_cpu,
    )

    return n_jobs
