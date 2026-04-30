"""Tests for the data processor."""

from datetime import UTC

import numpy as np
import pandas as pd

from tradingdev.domain.data.processor import DataProcessor


class TestDataProcessor:
    def setup_method(self) -> None:
        self.processor = DataProcessor()

    def test_sorts_by_timestamp(self) -> None:
        df = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(
                    ["2024-01-01 02:00", "2024-01-01 01:00", "2024-01-01 00:00"],
                    utc=True,
                ),
                "open": [1.0, 2.0, 3.0],
                "high": [1.0, 2.0, 3.0],
                "low": [1.0, 2.0, 3.0],
                "close": [1.0, 2.0, 3.0],
                "volume": [100.0, 200.0, 300.0],
            }
        )
        result = self.processor.process(df)
        assert result["timestamp"].is_monotonic_increasing

    def test_removes_duplicates(self) -> None:
        ts = pd.to_datetime("2024-01-01 00:00", utc=True)
        df = pd.DataFrame(
            {
                "timestamp": [ts, ts],
                "open": [1.0, 2.0],
                "high": [1.0, 2.0],
                "low": [1.0, 2.0],
                "close": [1.0, 2.0],
                "volume": [100.0, 200.0],
            }
        )
        result = self.processor.process(df)
        assert len(result) == 1

    def test_forward_fills_missing_ohlc(self) -> None:
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=3, freq="h", tz=UTC),
                "open": [1.0, np.nan, 3.0],
                "high": [1.0, np.nan, 3.0],
                "low": [1.0, np.nan, 3.0],
                "close": [1.0, np.nan, 3.0],
                "volume": [100.0, 200.0, 300.0],
            }
        )
        result = self.processor.process(df)
        assert result["open"].iloc[1] == 1.0  # forward-filled

    def test_zero_fills_missing_volume(self) -> None:
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=2, freq="h", tz=UTC),
                "open": [1.0, 2.0],
                "high": [1.0, 2.0],
                "low": [1.0, 2.0],
                "close": [1.0, 2.0],
                "volume": [100.0, np.nan],
            }
        )
        result = self.processor.process(df)
        assert result["volume"].iloc[1] == 0.0

    def test_ensures_utc(self) -> None:
        df = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(["2024-01-01 00:00"]),
                "open": [1.0],
                "high": [1.0],
                "low": [1.0],
                "close": [1.0],
                "volume": [100.0],
            }
        )
        result = self.processor.process(df)
        assert result["timestamp"].dt.tz is not None
