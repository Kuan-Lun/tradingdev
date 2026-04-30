"""Binance Data Vision batch downloader (skeleton for future implementation)."""

from datetime import datetime
from pathlib import Path

import pandas as pd

from tradingdev.domain.data.crawlers.base import BaseCrawler
from tradingdev.utils.logger import setup_logger

logger = setup_logger(__name__)


class BinanceVisionCrawler(BaseCrawler):
    """Batch download historical K-line data from data.binance.vision.

    This is a skeleton implementation. The full version will use httpx
    to download ZIP files containing daily/monthly K-line CSVs.
    """

    def fetch(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        """Fetch OHLCV data from Binance Data Vision.

        Not yet implemented — use ``BinanceAPICrawler`` for now.
        """
        raise NotImplementedError(
            "BinanceVisionCrawler.fetch() is not yet implemented. "
            "Use BinanceAPICrawler instead."
        )

    def save_raw(self, df: pd.DataFrame, output_path: Path) -> None:
        """Save raw data to disk."""
        raise NotImplementedError(
            "BinanceVisionCrawler.save_raw() is not yet implemented."
        )
