"""Shared test fixtures."""

from datetime import UTC

import numpy as np
import pandas as pd
import pytest


def _make_ohlcv(
    n: int,
    seed: int,
    start: str = "2024-01-01",
) -> pd.DataFrame:
    """Helper to build a synthetic OHLCV DataFrame."""
    rng = np.random.default_rng(seed=seed)

    timestamps = pd.date_range(
        start=start, periods=n, freq="h", tz=UTC
    )

    returns = rng.normal(0.0001, 0.005, size=n)
    close = 40_000.0 * np.cumprod(1 + returns)

    noise = rng.uniform(0.001, 0.005, size=n)
    high = close * (1 + noise)
    low = close * (1 - noise)
    open_ = close * (1 + rng.normal(0, 0.002, size=n))
    volume = rng.uniform(100, 1000, size=n)

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


@pytest.fixture
def sample_ohlcv_df() -> pd.DataFrame:
    """Create a synthetic OHLCV DataFrame with 100 rows of 1h BTC/USDT data.

    Generates a random walk price series around 40,000 with realistic
    OHLCV structure.
    """
    return _make_ohlcv(100, seed=42)


@pytest.fixture
def sample_ohlcv_with_kd() -> pd.DataFrame:
    """OHLCV data extended to 200 rows for KD indicator calculation.

    KD requires enough bars for the lookback period (default 14+3+3).
    """
    return _make_ohlcv(200, seed=123)


@pytest.fixture
def large_ohlcv_df() -> pd.DataFrame:
    """1000-row OHLCV DataFrame for ML strategy tests."""
    return _make_ohlcv(1000, seed=456)


@pytest.fixture
def walk_forward_ohlcv_df() -> pd.DataFrame:
    """500-row OHLCV DataFrame spanning ~20 days, for walk-forward tests."""
    return _make_ohlcv(500, seed=789)
