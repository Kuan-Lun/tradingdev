"""Storage location and identity for CLI pipeline result artifacts.

ArtifactService writes results and retrieves them by their registered run ID.
The cache key used when saving an artifact is derived from:

1. Executed manifest hash (includes effective settings and strategy defaults).
2. Processed data file **mtime + size** (catches data regeneration).
3. Git code fingerprint of ``src/`` (catches strategy logic changes).
   Falls back to a random value when git is unavailable, avoiding reuse of
   an unverified code identity when saving a result.
"""

from __future__ import annotations

import hashlib
import logging
import os
import subprocess
from pathlib import Path
from uuid import uuid4

from tradingdev.adapters.storage.filesystem import WorkspacePaths

logger = logging.getLogger(__name__)

CACHE_DIR: Path | None = None


def cache_dir() -> Path:
    """Return the workspace-aligned pipeline result cache directory."""
    if CACHE_DIR is not None:
        return CACHE_DIR
    data_root = os.environ.get("TRADINGDEV_DATA_ROOT")
    if data_root:
        return Path(data_root).expanduser().resolve() / "processed" / "cache"
    return WorkspacePaths().processed_data / "cache"


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
    function returns a random hex string so a new result does not reuse
    a cache key whose code identity could not be checked.
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
    *,
    manifest_hash: str,
    processed_path: Path,
) -> str:
    """Compute artifact identity from the executed manifest + data + code state."""
    h = hashlib.sha256()
    h.update(manifest_hash.encode("ascii"))
    if processed_path.exists():
        stat = processed_path.stat()
        h.update(f"{stat.st_mtime}:{stat.st_size}".encode())
    h.update(_code_fingerprint().encode())
    return h.hexdigest()[:16]


def clear_cache() -> int:
    """Remove all cached results. Returns number of files removed."""
    directory = cache_dir()
    if not directory.exists():
        return 0
    count = 0
    for f in directory.glob("*.pkl"):
        f.unlink()
        count += 1
    return count
