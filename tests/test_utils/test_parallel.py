"""Tests for parallel grid-search utilities."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd

from quant_backtest.utils.parallel import (
    _get_performance_core_count,
    estimate_n_jobs,
)


class TestGetPerformanceCoreCount:
    def test_returns_positive(self) -> None:
        assert _get_performance_core_count() >= 1

    def test_fallback_on_non_darwin(self) -> None:
        """On non-Darwin platforms, should fall back to os.cpu_count()."""
        with patch("quant_backtest.utils.parallel.platform") as mock_platform:
            mock_platform.system.return_value = "Linux"
            count = _get_performance_core_count()
            assert count >= 1

    def test_fallback_on_sysctl_failure(self) -> None:
        """When sysctl fails, should fall back to os.cpu_count()."""
        with (
            patch("quant_backtest.utils.parallel.platform") as mock_platform,
            patch("quant_backtest.utils.parallel.subprocess") as mock_subprocess,
        ):
            mock_platform.system.return_value = "Darwin"
            mock_subprocess.run.side_effect = OSError("sysctl not found")
            mock_subprocess.TimeoutExpired = TimeoutError
            count = _get_performance_core_count()
            assert count >= 1

    def test_apple_silicon_returns_pcore_count(self) -> None:
        """Simulate Apple Silicon sysctl returning P-core count."""
        with (
            patch("quant_backtest.utils.parallel.platform") as mock_platform,
            patch("quant_backtest.utils.parallel.subprocess") as mock_subprocess,
        ):
            mock_platform.system.return_value = "Darwin"
            mock_result = mock_subprocess.run.return_value
            mock_result.returncode = 0
            mock_result.stdout = "10\n"
            mock_subprocess.TimeoutExpired = TimeoutError
            assert _get_performance_core_count() == 10


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

    def test_does_not_exceed_perf_cores_minus_reserve(self) -> None:
        df = self._make_df()
        perf_cores = _get_performance_core_count()
        reserve = 2
        result = estimate_n_jobs(df, reserve_cores=reserve)
        assert result <= max(1, perf_cores - reserve)

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

    def test_reserve_cores_reduces_n_jobs(self) -> None:
        """More reserved cores should yield fewer workers."""
        df = self._make_df()
        n_low_reserve = estimate_n_jobs(df, reserve_cores=0)
        n_high_reserve = estimate_n_jobs(df, reserve_cores=4)
        assert n_high_reserve <= n_low_reserve

    def test_reserve_cores_never_below_one(self) -> None:
        """Even with very high reserve, result should be >= 1."""
        df = self._make_df()
        result = estimate_n_jobs(df, reserve_cores=9999)
        assert result >= 1
