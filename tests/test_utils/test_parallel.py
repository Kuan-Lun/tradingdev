"""Tests for parallel grid-search utilities."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from btc_strategy.utils.parallel import estimate_n_jobs


class TestEstimateNJobs:
    def _make_df(self, n_rows: int = 1000) -> pd.DataFrame:
        rng = np.random.default_rng(42)
        return pd.DataFrame(
            {
                "open": rng.uniform(90, 110, n_rows),
                "high": rng.uniform(95, 115, n_rows),
                "low": rng.uniform(85, 105, n_rows),
                "close": rng.uniform(90, 110, n_rows),
                "volume": rng.uniform(100, 10000, n_rows),
            }
        )

    def test_returns_at_least_one(self) -> None:
        df = self._make_df()
        assert estimate_n_jobs(df) >= 1

    def test_does_not_exceed_cpu_count(self) -> None:
        df = self._make_df()
        cpu_count = os.cpu_count() or 4
        assert estimate_n_jobs(df) <= cpu_count

    def test_large_df_reduces_n_jobs(self) -> None:
        """A very large DataFrame should yield fewer workers."""
        small_df = self._make_df(100)
        large_df = self._make_df(10_000_000)
        n_small = estimate_n_jobs(small_df)
        n_large = estimate_n_jobs(large_df)
        assert n_large <= n_small

    def test_custom_safety_factor(self) -> None:
        df = self._make_df()
        n_conservative = estimate_n_jobs(df, safety_factor=0.3)
        n_aggressive = estimate_n_jobs(df, safety_factor=0.9)
        assert n_conservative <= n_aggressive
