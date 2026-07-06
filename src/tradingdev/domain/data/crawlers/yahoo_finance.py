"""Yahoo Finance chart API crawler for stocks, futures, indices, and FX.

Uses the public v8 chart endpoint (no API key). Symbols follow Yahoo
conventions: ``AAPL`` (US stock), ``2330.TW`` (non-US stock), ``ES=F``
(futures), ``^GSPC`` (index), ``EURUSD=X`` (FX).

Yahoo Finance is an unofficial free API without a published quota;
intraday intervals are limited to roughly the last 730 days. Keep
request ranges small and rely on the yearly cache to avoid re-fetching.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import httpx
import pandas as pd

from tradingdev.domain.data.crawlers.base import BaseCrawler
from tradingdev.shared.utils.logger import setup_logger

if TYPE_CHECKING:
    from pathlib import Path

logger = setup_logger(__name__)

_BASE_URL = "https://query1.finance.yahoo.com/v8/finance/chart"
_OHLCV_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]
_USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"

_INTERVALS = {
    "1m": "1m",
    "2m": "2m",
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "1h": "60m",
    "60m": "60m",
    "90m": "90m",
    "1d": "1d",
    "1w": "1wk",
    "1wk": "1wk",
    "1mo": "1mo",
}


def _to_yahoo_interval(timeframe: str) -> str:
    """Map a project timeframe to a Yahoo chart interval."""
    interval = _INTERVALS.get(timeframe)
    if interval is None:
        supported = ", ".join(sorted(_INTERVALS))
        msg = (
            f"Timeframe {timeframe!r} is not supported by yahoo_finance. "
            f"Supported timeframes: {supported}"
        )
        raise ValueError(msg)
    return interval


def _parse_chart_payload(payload: dict[str, Any]) -> pd.DataFrame:
    """Convert a v8 chart JSON payload into a standard OHLCV DataFrame."""
    chart = payload.get("chart", {})
    error = chart.get("error")
    if error:
        msg = f"Yahoo Finance chart error: {error}"
        raise ValueError(msg)
    results = chart.get("result") or []
    if not results:
        return pd.DataFrame(columns=_OHLCV_COLUMNS)
    result = results[0]
    timestamps = result.get("timestamp") or []
    if not timestamps:
        return pd.DataFrame(columns=_OHLCV_COLUMNS)
    quote = result["indicators"]["quote"][0]
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(timestamps, unit="s", utc=True),
            "open": pd.to_numeric(pd.Series(quote["open"]), errors="coerce"),
            "high": pd.to_numeric(pd.Series(quote["high"]), errors="coerce"),
            "low": pd.to_numeric(pd.Series(quote["low"]), errors="coerce"),
            "close": pd.to_numeric(pd.Series(quote["close"]), errors="coerce"),
            "volume": pd.to_numeric(pd.Series(quote["volume"]), errors="coerce"),
        }
    )
    return df.dropna(subset=["close"]).reset_index(drop=True)


class YahooFinanceCrawler(BaseCrawler):
    """Fetch OHLCV data from the Yahoo Finance chart API (no API key)."""

    def __init__(self, timeout: float = 30.0) -> None:
        self._client = httpx.Client(
            timeout=timeout,
            follow_redirects=True,
            headers={"User-Agent": _USER_AGENT},
        )

    def fetch(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        """Fetch OHLCV candles for [start, end].

        Args:
            symbol: Yahoo Finance symbol (e.g. ``"AAPL"``, ``"ES=F"``).
            timeframe: Candle interval (e.g. ``"1d"``, ``"1h"``).
            start: Start time (UTC).
            end: End time (UTC).

        Returns:
            DataFrame with columns ``[timestamp, open, high, low, close, volume]``.
        """
        interval = _to_yahoo_interval(timeframe)
        start_utc = start if start.tzinfo else start.replace(tzinfo=UTC)
        end_utc = end if end.tzinfo else end.replace(tzinfo=UTC)
        now = datetime.now(UTC)
        if start_utc > now:
            return pd.DataFrame(columns=_OHLCV_COLUMNS)
        end_utc = min(end_utc, now)

        logger.info(
            "Fetching %s %s candles from %s to %s via Yahoo Finance",
            symbol,
            timeframe,
            start_utc.isoformat(),
            end_utc.isoformat(),
        )
        response = self._client.get(
            f"{_BASE_URL}/{symbol}",
            params={
                "period1": int(start_utc.timestamp()),
                "period2": int(end_utc.timestamp()),
                "interval": interval,
                "events": "history",
            },
        )
        response.raise_for_status()
        df = _parse_chart_payload(response.json())
        if df.empty:
            logger.warning("No data fetched for %s %s", symbol, timeframe)
            return df

        df = df[(df["timestamp"] >= start_utc) & (df["timestamp"] <= end_utc)]
        df = df.sort_values("timestamp").drop_duplicates("timestamp")
        logger.info("Total candles fetched: %d", len(df))
        return df.reset_index(drop=True)

    def save_raw(self, df: pd.DataFrame, output_path: Path) -> None:
        """Save raw OHLCV data as CSV.

        Args:
            df: OHLCV DataFrame.
            output_path: Path to write CSV file.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info("Saved raw data to %s", output_path)
