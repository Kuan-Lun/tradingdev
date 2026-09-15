"""Shared resolution of runtime paths across application and domain code."""

from __future__ import annotations

import os
from pathlib import Path


def resolve_workspace_root(root: Path | None = None) -> Path:
    """Resolve explicit workspace, environment override, then local default."""
    configured = os.environ.get("TRADINGDEV_WORKSPACE")
    selected = root if root is not None else Path(configured or "workspace")
    return selected.expanduser().resolve()
