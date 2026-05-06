"""Tests for the Binance Vision crawler (mocked HTTP responses)."""

from __future__ import annotations

import io
import zipfile
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from tradingdev.domain.data.crawlers.binance_vision import (
    BinanceVisionCrawler,
    _month_range,
    _parse_zip_csv,
    _to_vision_symbol,
)

if TYPE_CHECKING:
    from pathlib import Path

# ------------------------------------------------------------------ #
# Helper utilities                                                    #
# ------------------------------------------------------------------ #


def _make_zip_csv(rows: list[tuple[object, ...]]) -> bytes:
    """Build a fake Binance Vision ZIP with a single K-line CSV."""
    lines = [",".join(str(v) for v in row) for row in rows]
    csv_bytes = "\n".join(lines).encode()
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("BTCUSDT-1d-2024-01.csv", csv_bytes)
    return buf.getvalue()


def _candle_row(ts_ms: int) -> tuple[object, ...]:
    return (
        ts_ms,
        40000.0,
        40100.0,
        39900.0,
        40050.0,
        500.0,
        ts_ms + 86_399_999,
        1.0,
        10,
        1.0,
        1.0,
        0,
    )


def _jan_2024_zip() -> bytes:
    rows = [_candle_row(1704067200000 + i * 86_400_000) for i in range(31)]
    return _make_zip_csv(rows)


def _make_mock_response(content: bytes, status_code: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.content = content
    resp.raise_for_status = MagicMock()
    return resp


# ------------------------------------------------------------------ #
# Pure utility tests                                                  #
# ------------------------------------------------------------------ #


class TestToVisionSymbol:
    def test_slash_removed(self) -> None:
        assert _to_vision_symbol("BTC/USDT") == "BTCUSDT"

    def test_already_no_slash(self) -> None:
        assert _to_vision_symbol("ETHUSDT") == "ETHUSDT"

    def test_uppercased(self) -> None:
        assert _to_vision_symbol("btc/usdt") == "BTCUSDT"


class TestMonthRange:
    def test_single_month(self) -> None:
        start = datetime(2024, 3, 5, tzinfo=UTC)
        end = datetime(2024, 3, 28, tzinfo=UTC)
        assert _month_range(start, end) == [(2024, 3)]

    def test_two_months(self) -> None:
        start = datetime(2024, 1, 1, tzinfo=UTC)
        end = datetime(2024, 2, 28, tzinfo=UTC)
        assert _month_range(start, end) == [(2024, 1), (2024, 2)]

    def test_year_boundary(self) -> None:
        start = datetime(2024, 11, 1, tzinfo=UTC)
        end = datetime(2025, 2, 1, tzinfo=UTC)
        assert _month_range(start, end) == [
            (2024, 11),
            (2024, 12),
            (2025, 1),
            (2025, 2),
        ]


class TestParseZipCsv:
    def test_columns_and_types(self) -> None:
        ts_ms = 1704067200000
        content = _make_zip_csv([_candle_row(ts_ms)])
        df = _parse_zip_csv(content)
        assert list(df.columns) == [
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]
        assert df["timestamp"].dt.tz is not None
        assert df["open"].dtype == float

    def test_timestamp_value(self) -> None:
        ts_ms = 1704067200000
        content = _make_zip_csv([_candle_row(ts_ms)])
        df = _parse_zip_csv(content)
        expected = pd.Timestamp("2024-01-01", tz=UTC)
        assert df["timestamp"].iloc[0] == expected


# ------------------------------------------------------------------ #
# BinanceVisionCrawler integration tests (mocked HTTP)               #
# ------------------------------------------------------------------ #


class TestBinanceVisionCrawler:
    @pytest.fixture
    def crawler(self) -> BinanceVisionCrawler:
        return BinanceVisionCrawler(market_type="futures/um")

    def test_fetch_complete_past_month_uses_monthly_url(
        self, crawler: BinanceVisionCrawler
    ) -> None:
        zip_content = _jan_2024_zip()
        mock_resp = _make_mock_response(zip_content)

        with (
            patch.object(crawler._client, "get", return_value=mock_resp) as mock_get,
            patch("tradingdev.domain.data.crawlers.binance_vision.datetime") as mock_dt,
        ):
            mock_dt.now.return_value = datetime(2026, 5, 1, tzinfo=UTC)
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            df = crawler.fetch(
                "BTC/USDT",
                "1d",
                datetime(2024, 1, 1, tzinfo=UTC),
                datetime(2024, 1, 31, tzinfo=UTC),
            )

        called_url = mock_get.call_args[0][0]
        assert "monthly" in called_url
        assert "BTCUSDT-1d-2024-01.zip" in called_url
        assert len(df) == 31
        assert list(df.columns) == [
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]

    def test_fetch_filters_to_requested_range(
        self, crawler: BinanceVisionCrawler
    ) -> None:
        zip_content = _jan_2024_zip()
        mock_resp = _make_mock_response(zip_content)

        with (
            patch.object(crawler._client, "get", return_value=mock_resp),
            patch("tradingdev.domain.data.crawlers.binance_vision.datetime") as mock_dt,
        ):
            mock_dt.now.return_value = datetime(2026, 5, 1, tzinfo=UTC)
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            df = crawler.fetch(
                "BTC/USDT",
                "1d",
                datetime(2024, 1, 5, tzinfo=UTC),
                datetime(2024, 1, 10, tzinfo=UTC),
            )

        assert df["timestamp"].min() >= pd.Timestamp("2024-01-05", tz=UTC)
        assert df["timestamp"].max() <= pd.Timestamp("2024-01-10", tz=UTC)

    def test_fetch_404_returns_empty_dataframe(
        self, crawler: BinanceVisionCrawler
    ) -> None:
        mock_resp = _make_mock_response(b"", status_code=404)

        with (
            patch.object(crawler._client, "get", return_value=mock_resp),
            patch("tradingdev.domain.data.crawlers.binance_vision.datetime") as mock_dt,
        ):
            mock_dt.now.return_value = datetime(2026, 5, 1, tzinfo=UTC)
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            df = crawler.fetch(
                "BTC/USDT",
                "1d",
                datetime(2024, 1, 1, tzinfo=UTC),
                datetime(2024, 1, 31, tzinfo=UTC),
            )

        assert df.empty
        assert list(df.columns) == [
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]

    def test_market_type_in_url(self, crawler: BinanceVisionCrawler) -> None:
        zip_content = _jan_2024_zip()
        mock_resp = _make_mock_response(zip_content)

        with (
            patch.object(crawler._client, "get", return_value=mock_resp) as mock_get,
            patch("tradingdev.domain.data.crawlers.binance_vision.datetime") as mock_dt,
        ):
            mock_dt.now.return_value = datetime(2026, 5, 1, tzinfo=UTC)
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            crawler.fetch(
                "BTC/USDT",
                "1d",
                datetime(2024, 1, 1, tzinfo=UTC),
                datetime(2024, 1, 31, tzinfo=UTC),
            )

        called_url = mock_get.call_args[0][0]
        assert "futures/um" in called_url

    def test_spot_market_type_in_url(self) -> None:
        spot_crawler = BinanceVisionCrawler(market_type="spot")
        zip_content = _jan_2024_zip()
        mock_resp = _make_mock_response(zip_content)

        with (
            patch.object(
                spot_crawler._client, "get", return_value=mock_resp
            ) as mock_get,
            patch("tradingdev.domain.data.crawlers.binance_vision.datetime") as mock_dt,
        ):
            mock_dt.now.return_value = datetime(2026, 5, 1, tzinfo=UTC)
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            spot_crawler.fetch(
                "BTC/USDT",
                "1d",
                datetime(2024, 1, 1, tzinfo=UTC),
                datetime(2024, 1, 31, tzinfo=UTC),
            )

        called_url = mock_get.call_args[0][0]
        assert "spot" in called_url

    def test_save_raw_creates_csv(
        self, crawler: BinanceVisionCrawler, tmp_path: Path
    ) -> None:
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=3, freq="D", tz=UTC),
                "open": [1.0, 2.0, 3.0],
                "high": [1.0, 2.0, 3.0],
                "low": [1.0, 2.0, 3.0],
                "close": [1.0, 2.0, 3.0],
                "volume": [100.0, 200.0, 300.0],
            }
        )
        output = tmp_path / "test.csv"
        crawler.save_raw(df, output)
        assert output.exists()
