"""Persistent disk cache for PipelineResult objects.

Avoids re-running expensive backtests when the config, data, and
source code have not changed.  The cache key is derived from:

1. YAML config file **content** (catches any parameter change).
2. Processed data file **mtime + size** (catches data regeneration).
3. Git code fingerprint of ``src/`` (catches strategy logic changes).
   Falls back to a random value when git is unavailable, ensuring
   no stale cache is used.
"""

from __future__ import annotations

import hashlib
import logging
import pickle
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

if TYPE_CHECKING:
    from tradingdev.backtest.pipeline_result import (
        PipelineResult,
    )

logger = logging.getLogger(__name__)

CACHE_DIR = Path("data/cache")


def _run_git(*args: str, cwd: Path) -> str | None:
    """Run a git command and return stdout, or *None* on failure."""
    try:
        proc = subprocess.run(  # noqa: S603, S607
            ["git", *args],
            capture_output=True,
            text=True,
            cwd=cwd,
            check=False,
        )
    except FileNotFoundError:
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout


def _code_fingerprint() -> str:
    """Derive a fingerprint for the current state of ``src/``.

    Combines:
    * ``git rev-parse HEAD`` — committed code state.
    * ``git diff HEAD -- src/`` — uncommitted changes (staged + unstaged).
    * Content of untracked files under ``src/``.

    Returns a 16-char hex digest.  If any git command fails the
    function returns a random hex string so the cache is always
    invalidated (safe fallback).
    """
    # Locate the repository root.
    toplevel = _run_git("rev-parse", "--show-toplevel", cwd=Path.cwd())
    if toplevel is None:
        logger.debug("git not available; using random code fingerprint")
        return uuid4().hex[:16]

    repo_root = Path(toplevel.strip())
    h = hashlib.sha256()

    # 1) HEAD commit hash.
    commit = _run_git("rev-parse", "HEAD", cwd=repo_root)
    if commit is None:
        return uuid4().hex[:16]
    h.update(commit.strip().encode())

    # 2) Uncommitted changes in src/ (staged + unstaged).
    diff = _run_git("diff", "HEAD", "--", "src/", cwd=repo_root)
    if diff is not None:
        h.update(diff.encode())

    # 3) Untracked files in src/.
    untracked = _run_git(
        "ls-files",
        "--others",
        "--exclude-standard",
        "src/",
        cwd=repo_root,
    )
    if untracked:
        for rel in sorted(untracked.strip().splitlines()):
            filepath = repo_root / rel
            if filepath.is_file():
                h.update(filepath.read_bytes())

    return h.hexdigest()[:16]


def compute_cache_key(
    config_path: Path,
    processed_path: Path,
) -> str:
    """Compute a SHA-256 cache key from config + data + code state."""
    h = hashlib.sha256()
    h.update(config_path.read_bytes())
    if processed_path.exists():
        stat = processed_path.stat()
        h.update(f"{stat.st_mtime}:{stat.st_size}".encode())
    h.update(_code_fingerprint().encode())
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
