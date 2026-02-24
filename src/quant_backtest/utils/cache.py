"""Persistent disk cache for PipelineResult objects.

Avoids re-running expensive backtests when the config and data
have not changed.  The cache key is derived from:

1. YAML config file **content** (catches any parameter change).
2. Processed data file **mtime + size** (catches data regeneration).

Code changes are *not* tracked — delete ``data/cache/`` manually
after modifying strategy logic.
"""

from __future__ import annotations

import hashlib
import logging
import pickle
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from quant_backtest.backtest.pipeline_result import (
        PipelineResult,
    )

logger = logging.getLogger(__name__)

CACHE_DIR = Path("data/cache")


def compute_cache_key(
    config_path: Path,
    processed_path: Path,
) -> str:
    """Compute a SHA-256 cache key from config + data metadata."""
    h = hashlib.sha256()
    h.update(config_path.read_bytes())
    if processed_path.exists():
        stat = processed_path.stat()
        h.update(f"{stat.st_mtime}:{stat.st_size}".encode())
    return h.hexdigest()[:16]


def load_cached_result(
    config_path: Path,
    processed_path: Path,
) -> PipelineResult | None:
    """Load a cached PipelineResult if one exists for this key."""
    key = compute_cache_key(config_path, processed_path)
    cache_file = CACHE_DIR / f"{key}.pkl"
    if not cache_file.exists():
        return None
    try:
        with open(cache_file, "rb") as f:
            result: PipelineResult = pickle.load(f)  # noqa: S301
        logger.info("Loaded cached result from %s", cache_file.name)
        return result
    except Exception:
        logger.warning(
            "Failed to load cache %s, will re-run",
            cache_file.name,
        )
        return None


def save_cached_result(
    result: PipelineResult,
    config_path: Path,
    processed_path: Path,
) -> Path:
    """Persist a PipelineResult to disk. Returns the cache file path."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = compute_cache_key(config_path, processed_path)
    cache_file = CACHE_DIR / f"{key}.pkl"
    with open(cache_file, "wb") as f:
        pickle.dump(result, f)
    logger.info("Saved result cache to %s", cache_file.name)
    return cache_file


def clear_cache() -> int:
    """Remove all cached results. Returns number of files removed."""
    if not CACHE_DIR.exists():
        return 0
    count = 0
    for f in CACHE_DIR.glob("*.pkl"):
        f.unlink()
        count += 1
    return count
