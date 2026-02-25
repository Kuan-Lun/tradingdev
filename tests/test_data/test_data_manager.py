"""Tests for the yearly-cached DataManager."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from quant_backtest.data.data_manager import (
    DataManager,
    _is_year_complete,
    _normalize_symbol,
    _year_range,
)
from quant_backtest.data.schemas import BacktestConfig, DataConfig

# ------------------------------------------------------------------ #
# Utility function tests                                              #
# ------------------------------------------------------------------ #


class TestNormalizeSymbol:
    def test_btc_usdt(self) -> None:
        assert _normalize_symbol("BTC/USDT") == "btcusdt"

    def test_eth_usdt(self) -> None:
        assert _normalize_symbol("ETH/USDT") == "ethusdt"

    def test_already_normalized(self) -> None:
        assert _normalize_symbol("btcusdt") == "btcusdt"


class TestYearRange:
    def test_single_year(self) -> None:
        start = datetime(2024, 3, 1, tzinfo=UTC)
        end = datetime(2024, 11, 30, tzinfo=UTC)
        assert _year_range(start, end) == [2024]

    def test_two_years(self) -> None:
        start = datetime(2024, 1, 1, tzinfo=UTC)
        end = datetime(2025, 12, 31, tzinfo=UTC)
        assert _year_range(start, end) == [2024, 2025]

    def test_three_years(self) -> None:
        start = datetime(2023, 6, 1, tzinfo=UTC)
        end = datetime(2025, 3, 1, tzinfo=UTC)
        assert _year_range(start, end) == [2023, 2024, 2025]


class TestIsYearComplete:
    def test_past_year(self) -> None:
        now_fn = lambda: datetime(2026, 2, 1, tzinfo=UTC)  # noqa: E731
        assert _is_year_complete(2025, now_fn) is True

    def test_current_year(self) -> None:
        now_fn = lambda: datetime(2026, 2, 1, tzinfo=UTC)  # noqa: E731
        assert _is_year_complete(2026, now_fn) is False

    def test_boundary_before_year_end(self) -> None:
        now_fn = lambda: datetime(2024, 12, 31, 23, 59, 58, tzinfo=UTC)  # noqa: E731
        assert _is_year_complete(2024, now_fn) is False

    def test_boundary_after_year_end(self) -> None:
        now_fn = lambda: datetime(2025, 1, 1, 0, 0, 0, tzinfo=UTC)  # noqa: E731
        assert _is_year_complete(2024, now_fn) is True


# ------------------------------------------------------------------ #
# Path generation tests                                               #
# ------------------------------------------------------------------ #


class TestPathGeneration:
    def setup_method(self) -> None:
        self.data_cfg = DataConfig(source="binance_api")
        self.bt_cfg = BacktestConfig(
            symbol="BTC/USDT",
            timeframe="1h",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 12, 31),
        )
        self.manager = DataManager(
            data_config=self.data_cfg,
            backtest_config=self.bt_cfg,
        )

    def test_raw_path_complete(self) -> None:
        path = self.manager._raw_path_for_year(2024, partial=False)
        assert path == Path("data/raw/btcusdt_1h_2024.csv")

    def test_raw_path_partial(self) -> None:
        path = self.manager._raw_path_for_year(2026, partial=True)
        assert path == Path("data/raw/btcusdt_1h_2026_partial.csv")

    def test_processed_path_complete(self) -> None:
        path = self.manager._processed_path_for_year(2024, partial=False)
        assert path == Path("data/processed/btcusdt_1h_2024.parquet")

    def test_processed_path_partial(self) -> None:
        path = self.manager._processed_path_for_year(2026, partial=True)
        assert path == Path("data/processed/btcusdt_1h_2026_partial.parquet")

    def test_custom_dirs(self) -> None:
        data_cfg = DataConfig(
            source="binance_api",
            raw_dir="custom/raw",
            processed_dir="custom/processed",
        )
        manager = DataManager(data_config=data_cfg, backtest_config=self.bt_cfg)
        assert manager._raw_path_for_year(2024) == Path(
            "custom/raw/btcusdt_1h_2024.csv"
        )
        assert manager._processed_path_for_year(2024) == Path(
            "custom/processed/btcusdt_1h_2024.parquet"
        )


# ------------------------------------------------------------------ #
# Yearly cache logic tests                                            #
# ------------------------------------------------------------------ #


class TestDataManagerYearly:
    """Test yearly caching logic with mocked crawler."""

    @pytest.fixture
    def _dirs(self, tmp_path: Path) -> tuple[Path, Path]:
        """Create raw and processed directories under tmp_path."""
        raw_dir = tmp_path / "raw"
        proc_dir = tmp_path / "processed"
        raw_dir.mkdir()
        proc_dir.mkdir()
        return raw_dir, proc_dir

    def test_single_complete_year(
        self,
        sample_ohlcv_df: pd.DataFrame,
        _dirs: tuple[Path, Path],
    ) -> None:
        """Fetch a single complete year and verify it is cached."""
        raw_dir, proc_dir = _dirs
        data_cfg = DataConfig(
            source="binance_api",
            raw_dir=str(raw_dir),
            processed_dir=str(proc_dir),
        )
        bt_cfg = BacktestConfig(
            symbol="BTC/USDT",
            timeframe="1h",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 12, 31),
        )
        frozen_now = lambda: datetime(2026, 2, 1, tzinfo=UTC)  # noqa: E731
        manager = DataManager(
            data_config=data_cfg,
            backtest_config=bt_cfg,
            now_fn=frozen_now,
        )

        with (
            patch.object(manager._crawler, "fetch", return_value=sample_ohlcv_df),
            patch.object(manager._crawler, "save_raw"),
        ):
            df, _ = manager.load()

        assert len(df) > 0
        expected = proc_dir / "btcusdt_1h_2024.parquet"
        assert expected.exists()

    def test_partial_year_creates_partial_file(
        self,
        sample_ohlcv_df: pd.DataFrame,
        _dirs: tuple[Path, Path],
    ) -> None:
        """Current year should create _partial file."""
        raw_dir, proc_dir = _dirs
        data_cfg = DataConfig(
            source="binance_api",
            raw_dir=str(raw_dir),
            processed_dir=str(proc_dir),
        )
        bt_cfg = BacktestConfig(
            symbol="BTC/USDT",
            timeframe="1h",
            start_date=datetime(2026, 1, 1),
            end_date=datetime(2026, 12, 31),
        )
        frozen_now = lambda: datetime(2026, 6, 15, tzinfo=UTC)  # noqa: E731
        manager = DataManager(
            data_config=data_cfg,
            backtest_config=bt_cfg,
            now_fn=frozen_now,
        )

        with (
            patch.object(manager._crawler, "fetch", return_value=sample_ohlcv_df),
            patch.object(manager._crawler, "save_raw"),
        ):
            manager.load()

        expected_partial = proc_dir / "btcusdt_1h_2026_partial.parquet"
        assert expected_partial.exists()
        # Complete file should NOT exist
        assert not (proc_dir / "btcusdt_1h_2026.parquet").exists()

    def test_partial_upgraded_to_complete(
        self,
        sample_ohlcv_df: pd.DataFrame,
        _dirs: tuple[Path, Path],
    ) -> None:
        """When year completes, partial should be deleted and re-fetched."""
        raw_dir, proc_dir = _dirs
        partial_path = proc_dir / "btcusdt_1h_2025_partial.parquet"
        sample_ohlcv_df.to_parquet(partial_path, index=False)

        data_cfg = DataConfig(
            source="binance_api",
            raw_dir=str(raw_dir),
            processed_dir=str(proc_dir),
        )
        bt_cfg = BacktestConfig(
            symbol="BTC/USDT",
            timeframe="1h",
            start_date=datetime(2025, 1, 1),
            end_date=datetime(2025, 12, 31),
        )
        # Now it's 2026 — 2025 is complete
        frozen_now = lambda: datetime(2026, 2, 1, tzinfo=UTC)  # noqa: E731
        manager = DataManager(
            data_config=data_cfg,
            backtest_config=bt_cfg,
            now_fn=frozen_now,
        )

        with (
            patch.object(manager._crawler, "fetch", return_value=sample_ohlcv_df),
            patch.object(manager._crawler, "save_raw"),
        ):
            manager.load()

        # Partial should be deleted
        assert not partial_path.exists()
        # Complete should exist
        complete_path = proc_dir / "btcusdt_1h_2025.parquet"
        assert complete_path.exists()

    def test_complete_file_reused(
        self,
        sample_ohlcv_df: pd.DataFrame,
        _dirs: tuple[Path, Path],
    ) -> None:
        """When complete file exists, crawler should NOT be called."""
        _, proc_dir = _dirs
        complete_path = proc_dir / "btcusdt_1h_2024.parquet"
        sample_ohlcv_df.to_parquet(complete_path, index=False)

        data_cfg = DataConfig(
            source="binance_api",
            processed_dir=str(proc_dir),
        )
        bt_cfg = BacktestConfig(
            symbol="BTC/USDT",
            timeframe="1h",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 12, 31),
        )
        frozen_now = lambda: datetime(2026, 2, 1, tzinfo=UTC)  # noqa: E731
        manager = DataManager(
            data_config=data_cfg,
            backtest_config=bt_cfg,
            now_fn=frozen_now,
        )

        with patch.object(manager._crawler, "fetch") as mock_fetch:
            manager.load()

        mock_fetch.assert_not_called()

    def test_multi_year_concat_and_trim(
        self,
        sample_ohlcv_df: pd.DataFrame,
        _dirs: tuple[Path, Path],
    ) -> None:
        """Multi-year: load 2024 + 2025, trim to exact range."""
        _, proc_dir = _dirs

        # Pre-populate cached files for both years
        for year in [2024, 2025]:
            path = proc_dir / f"btcusdt_1h_{year}.parquet"
            sample_ohlcv_df.to_parquet(path, index=False)

        data_cfg = DataConfig(
            source="binance_api",
            processed_dir=str(proc_dir),
        )
        bt_cfg = BacktestConfig(
            symbol="BTC/USDT",
            timeframe="1h",
            start_date=datetime(2024, 1, 2),
            end_date=datetime(2024, 1, 3),
        )
        frozen_now = lambda: datetime(2026, 2, 1, tzinfo=UTC)  # noqa: E731
        manager = DataManager(
            data_config=data_cfg,
            backtest_config=bt_cfg,
            now_fn=frozen_now,
        )

        df, _ = manager.load()
        # Verify trimming was applied (original data starts 2024-01-01)
        assert df["timestamp"].min() >= pd.Timestamp("2024-01-02", tz=UTC)
        assert df["timestamp"].max() <= pd.Timestamp("2024-01-03", tz=UTC)

    def test_partial_year_reused_when_still_incomplete(
        self,
        sample_ohlcv_df: pd.DataFrame,
        _dirs: tuple[Path, Path],
    ) -> None:
        """Partial file should be reused when year is still incomplete."""
        _, proc_dir = _dirs
        partial_path = proc_dir / "btcusdt_1h_2026_partial.parquet"
        sample_ohlcv_df.to_parquet(partial_path, index=False)

        data_cfg = DataConfig(
            source="binance_api",
            processed_dir=str(proc_dir),
        )
        bt_cfg = BacktestConfig(
            symbol="BTC/USDT",
            timeframe="1h",
            start_date=datetime(2026, 1, 1),
            end_date=datetime(2026, 12, 31),
        )
        # Still 2026 — year not complete
        frozen_now = lambda: datetime(2026, 6, 15, tzinfo=UTC)  # noqa: E731
        manager = DataManager(
            data_config=data_cfg,
            backtest_config=bt_cfg,
            now_fn=frozen_now,
        )

        with patch.object(manager._crawler, "fetch") as mock_fetch:
            manager.load()

        # Should NOT re-fetch, just use existing partial
        mock_fetch.assert_not_called()
