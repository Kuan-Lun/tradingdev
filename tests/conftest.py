"""Shared test fixtures."""

from datetime import UTC

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_ohlcv_df() -> pd.DataFrame:
    """Create a synthetic OHLCV DataFrame with 100 rows of 1h BTC/USDT data.

    Generates a random walk price series around 40,000 with realistic
    OHLCV structure.
    """
    rng = np.random.default_rng(seed=42)
    n = 100

    timestamps = pd.date_range(start="2024-01-01", periods=n, freq="h", tz=UTC)

    # Random walk for close prices
    returns = rng.normal(0.0001, 0.005, size=n)
    close = 40_000.0 * np.cumprod(1 + returns)

    # Derive OHLV from close
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
def sample_ohlcv_with_kd(sample_ohlcv_df: pd.DataFrame) -> pd.DataFrame:
    """OHLCV data extended to 200 rows for KD indicator calculation.

    KD requires enough bars for the lookback period (default 14+3+3).
    """
    rng = np.random.default_rng(seed=123)
    n = 200

    timestamps = pd.date_range(start="2024-01-01", periods=n, freq="h", tz=UTC)

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
