"""Binance public API crawler using ccxt."""

import time
from datetime import UTC, datetime
from pathlib import Path

import ccxt
import pandas as pd

from tradingdev.domain.data.crawlers.base import BaseCrawler
from tradingdev.utils.logger import setup_logger

logger = setup_logger(__name__)

_OHLCV_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]
_BATCH_LIMIT = 1000


class BinanceAPICrawler(BaseCrawler):
    """Fetch OHLCV data from Binance via ccxt (no API key required)."""

    def __init__(self) -> None:
        self._exchange: ccxt.binance = ccxt.binance({"enableRateLimit": True})

    def fetch(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        """Fetch OHLCV candles with automatic pagination.

        Args:
            symbol: Trading pair (e.g. ``"BTC/USDT"``).
            timeframe: Candle interval (e.g. ``"1h"``).
            start: Start time (UTC).
            end: End time (UTC).

        Returns:
            DataFrame with columns ``[timestamp, open, high, low, close, volume]``.
        """
        since_ms = int(start.timestamp() * 1000)
        end_ms = int(end.timestamp() * 1000)
        all_candles: list[list[object]] = []

        logger.info(
            "Fetching %s %s candles from %s to %s",
            symbol,
            timeframe,
            start.isoformat(),
            end.isoformat(),
        )

        while since_ms < end_ms:
            candles: list[list[object]] = self._exchange.fetch_ohlcv(
                symbol, timeframe, since=since_ms, limit=_BATCH_LIMIT
            )
            if not candles:
                break

            all_candles.extend(candles)
            last_ts = int(candles[-1][0])  # type: ignore[call-overload]
            logger.info(
                "Fetched %d candles, last timestamp: %s",
                len(candles),
                datetime.fromtimestamp(last_ts / 1000, tz=UTC).isoformat(),
            )

            if len(candles) < _BATCH_LIMIT:
                break

            since_ms = last_ts + 1
            time.sleep(self._exchange.rateLimit / 1000)

        df = pd.DataFrame(all_candles, columns=_OHLCV_COLUMNS)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)

        # Filter to requested range
        start_utc = start.replace(tzinfo=UTC)
        end_utc = end.replace(tzinfo=UTC)
        df = df[(df["timestamp"] >= start_utc) & (df["timestamp"] <= end_utc)]
        df = df.reset_index(drop=True)

        logger.info("Total candles fetched: %d", len(df))
        return df

    def save_raw(self, df: pd.DataFrame, output_path: Path) -> None:
        """Save raw OHLCV data as CSV.

        Args:
            df: OHLCV DataFrame.
            output_path: Path to write CSV file.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info("Saved raw data to %s", output_path)
