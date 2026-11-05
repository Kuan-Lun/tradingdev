"""Shared rolling prediction with automatic or named causal geometry."""

from __future__ import annotations

from tradingdev.domain.ml.rolling_gen.config import (
    RollingGenConfig,
    RollingSchemeConfig,
)
from tradingdev.domain.ml.rolling_gen.engine import (
    RollingOutcome,
    RollTrace,
    UnavailableReason,
    rolling_outcome,
    rolling_probabilities,
    run_rolling,
    signals_from_probabilities,
    trace_payload,
)
from tradingdev.domain.ml.rolling_gen.registry import (
    ROLLING_SCHEMES,
    get_rolling_scheme,
    resolve_rolling_scheme,
    rolling_scheme_names,
)
from tradingdev.domain.ml.rolling_gen.schedule import RollWindow, enumerate_rolls
from tradingdev.domain.ml.rolling_gen.scheme import (
    RollingGeometry,
    RollingScheme,
    RollType,
    resolve_geometry,
)

__all__ = [
    "ROLLING_SCHEMES",
    "RollTrace",
    "RollType",
    "RollWindow",
    "UnavailableReason",
    "RollingGenConfig",
    "RollingGeometry",
    "RollingOutcome",
    "RollingScheme",
    "RollingSchemeConfig",
    "enumerate_rolls",
    "get_rolling_scheme",
    "resolve_geometry",
    "resolve_rolling_scheme",
    "rolling_outcome",
    "rolling_probabilities",
    "rolling_scheme_names",
    "run_rolling",
    "signals_from_probabilities",
    "trace_payload",
]
