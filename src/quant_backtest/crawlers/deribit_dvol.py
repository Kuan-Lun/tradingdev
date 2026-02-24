"""Deribit DVOL (implied volatility index) crawler."""

import time
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pandas as pd

from quant_backtest.crawlers.base import BaseCrawler
from quant_backtest.utils.logger import setup_logger

logger = setup_logger(__name__)

_DVOL_COLUMNS = [
    "timestamp",
    "dvol_open",
    "dvol_high",
    "dvol_low",
    "dvol_close",
]
_BASE_URL = "https://www.deribit.com/api/v2/public/get_volatility_index_data"
_REQUEST_DELAY = 0.5  # seconds between paginated requests

# Deribit resolution parameter (seconds): 1m = "60"
_TIMEFRAME_MAP: dict[str, str] = {
    "1m": "60",
    "1h": "3600",
    "12h": "43200",
    "1d": "1D",
}


class DeribitDVOLCrawler(BaseCrawler):
    """Fetch Deribit DVOL data via public API (no key required).

    The DVOL index measures the 30-day implied volatility of BTC
    derived from Deribit's options market, analogous to the VIX.
    Values are annualized volatility in percent (e.g. 45 = 45%).
    """

    def __init__(self) -> None:
        self._client = httpx.Client(timeout=30.0)

    def fetch(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        """Fetch DVOL candles with automatic pagination.

        Args:
            symbol: Currency (e.g. ``"BTC"``).
            timeframe: Resolution (``"1m"``, ``"1h"``, ``"1d"``).
            start: Start time (UTC).
            end: End time (UTC).

        Returns:
            DataFrame with columns
            ``[timestamp, dvol_open, dvol_high, dvol_low, dvol_close]``.
        """
        resolution = _TIMEFRAME_MAP.get(timeframe)
        if resolution is None:
            msg = (
                f"Unsupported timeframe '{timeframe}'. "
                f"Supported: {list(_TIMEFRAME_MAP)}"
            )
            raise ValueError(msg)

        start_ms = int(start.timestamp() * 1000)
        end_ms = int(end.timestamp() * 1000)
        all_rows: list[list[float]] = []

        logger.info(
            "Fetching %s DVOL (%s) from %s to %s",
            symbol,
            timeframe,
            start.isoformat(),
            end.isoformat(),
        )

        current_end_ms = end_ms
        while current_end_ms > start_ms:
            params: dict[str, str | int] = {
                "currency": symbol.upper(),
                "start_timestamp": start_ms,
                "end_timestamp": current_end_ms,
                "resolution": resolution,
            }
            resp = self._client.get(_BASE_URL, params=params)
            resp.raise_for_status()
            body = resp.json()

            data: list[list[float]] = body["result"]["data"]
            if not data:
                break

            all_rows.extend(data)
            continuation = body["result"].get("continuation")

            logger.info(
                "Fetched %d DVOL candles, total so far: %d",
                len(data),
                len(all_rows),
            )

            if continuation is None:
                break

            current_end_ms = int(continuation)
            time.sleep(_REQUEST_DELAY)

        if not all_rows:
            logger.warning("No DVOL data returned for %s", symbol)
            return pd.DataFrame(columns=_DVOL_COLUMNS)

        df = pd.DataFrame(all_rows, columns=_DVOL_COLUMNS)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)

        # Sort chronologically and remove duplicates
        df = df.sort_values("timestamp").drop_duplicates(
            subset=["timestamp"], keep="first"
        )

        # Filter to requested range
        start_utc = start.replace(tzinfo=UTC)
        end_utc = end.replace(tzinfo=UTC)
        df = df[(df["timestamp"] >= start_utc) & (df["timestamp"] <= end_utc)]
        df = df.reset_index(drop=True)

        logger.info("Total DVOL candles fetched: %d", len(df))
        return df

    def save_raw(self, df: pd.DataFrame, output_path: Path) -> None:
        """Save raw DVOL data as CSV.

        Args:
            df: DVOL DataFrame.
            output_path: Path to write CSV file.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info("Saved raw DVOL data to %s", output_path)
