"""Yearly-cached data manager for OHLCV data acquisition.

Fetches, caches, and assembles OHLCV data at yearly granularity.
Partial years (current or incomplete) are cached separately and
re-fetched once the year completes.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from tradingdev.domain.data.crawlers.binance_api import BinanceAPICrawler
from tradingdev.domain.data.crawlers.binance_vision import BinanceVisionCrawler
from tradingdev.domain.data.loader import DataLoader
from tradingdev.domain.data.processor import DataProcessor

if TYPE_CHECKING:
    from collections.abc import Callable

    from tradingdev.domain.backtest.schemas import BacktestConfig
    from tradingdev.domain.data.crawlers.base import BaseCrawler
    from tradingdev.domain.data.schemas import DataConfig

logger = logging.getLogger(__name__)


def _now_utc() -> datetime:
    """Return the current UTC time."""
    return datetime.now(UTC)


def _normalize_symbol(symbol: str) -> str:
    """Normalize symbol for file naming: ``'BTC/USDT'`` -> ``'btcusdt'``."""
    return symbol.replace("/", "").lower()


def _is_year_complete(year: int, now_fn: Callable[[], datetime] = _now_utc) -> bool:
    """Check if a calendar year has fully elapsed."""
    year_end = datetime(year, 12, 31, 23, 59, 59, tzinfo=UTC)
    return now_fn() > year_end


def _year_range(start_date: datetime, end_date: datetime) -> list[int]:
    """Return list of calendar years covered by ``[start_date, end_date]``."""
    return list(range(start_date.year, end_date.year + 1))


def _year_start(year: int) -> datetime:
    """Return Jan 1 00:00:00 UTC of the given year."""
    return datetime(year, 1, 1, tzinfo=UTC)


def _year_end(year: int) -> datetime:
    """Return Dec 31 23:59:59 UTC of the given year."""
    return datetime(year, 12, 31, 23, 59, 59, tzinfo=UTC)


class DataManager:
    """Yearly-cached OHLCV data manager.

    Orchestrates data fetching, processing, and caching at yearly
    granularity.  Each calendar year is stored as a separate file.
    Incomplete (current) years are marked ``_partial`` and re-fetched
    once the year ends.

    Args:
        data_config: Parsed data section configuration.
        backtest_config: Parsed backtest section configuration.
        now_fn: Callable returning the current UTC datetime.
                Defaults to ``_now_utc``.  Override for testing.
    """

    def __init__(
        self,
        data_config: DataConfig,
        backtest_config: BacktestConfig,
        *,
        now_fn: Callable[[], datetime] = _now_utc,
    ) -> None:
        self._data_cfg = data_config
        self._bt_cfg = backtest_config
        self._now_fn = now_fn
        self._crawler: BaseCrawler = self._build_crawler()
        self._loader = DataLoader()
        self._processor = DataProcessor()

    def _build_crawler(self) -> BaseCrawler:
        if self._data_cfg.source == "binance_api":
            return BinanceAPICrawler()
        return BinanceVisionCrawler(market_type=self._data_cfg.market_type)

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #

    def load(self) -> tuple[pd.DataFrame, Path]:
        """Load data, fetching and caching as needed.

        Returns:
            A tuple of (trimmed DataFrame, effective processed path).
            The effective path is the first year's processed file and
            can be used for cache key computation.
        """
        years = _year_range(self._bt_cfg.start_date, self._bt_cfg.end_date)
        logger.info(
            "Yearly cache mode: years=%s, range=%s to %s",
            years,
            self._bt_cfg.start_date.isoformat(),
            self._bt_cfg.end_date.isoformat(),
        )

        yearly_dfs: list[pd.DataFrame] = []
        for year in years:
            df = self._ensure_year_cached(year)
            yearly_dfs.append(df)

        combined = pd.concat(yearly_dfs, ignore_index=True)
        combined = combined.sort_values("timestamp").reset_index(drop=True)
        combined = self._trim_to_range(combined)

        logger.info("Combined data: %d rows after trimming", len(combined))
        return combined, self.effective_processed_path

    @property
    def effective_processed_path(self) -> Path:
        """Return the processed path used for cache key computation.

        Returns the first year's processed file path.
        """
        years = _year_range(self._bt_cfg.start_date, self._bt_cfg.end_date)
        year = years[0]
        complete = _is_year_complete(year, self._now_fn)
        return self._processed_path_for_year(year, partial=not complete)

    # ------------------------------------------------------------------ #
    # Yearly cache logic                                                  #
    # ------------------------------------------------------------------ #

    def _ensure_year_cached(self, year: int) -> pd.DataFrame:
        """Ensure a single year's data is cached, fetching if needed.

        Logic:
            1. Complete-year parquet exists → load it.
            2. Year is now complete but only partial exists
               → delete partial, re-fetch complete.
            3. Year is still incomplete and partial exists
               → use partial.
            4. Nothing exists → fetch and save.
        """
        complete_path = self._processed_path_for_year(year, partial=False)
        partial_path = self._processed_path_for_year(year, partial=True)
        year_complete = _is_year_complete(year, self._now_fn)

        # Case 1: complete file exists
        if complete_path.exists():
            logger.info("Loading cached complete year %d", year)
            return self._loader.load_parquet(complete_path)

        # Case 2: year is now complete but only partial file exists
        if year_complete and partial_path.exists():
            logger.info(
                "Year %d is now complete; removing partial and re-fetching",
                year,
            )
            partial_path.unlink()
            raw_partial = self._raw_path_for_year(year, partial=True)
            if raw_partial.exists():
                raw_partial.unlink()

        # Case 3: year is still incomplete and partial exists
        if not year_complete and partial_path.exists():
            logger.info("Loading cached partial year %d", year)
            return self._loader.load_parquet(partial_path)

        # Case 4: need to fetch
        return self._fetch_and_cache_year(year, year_complete)

    def _fetch_and_cache_year(self, year: int, year_complete: bool) -> pd.DataFrame:
        """Fetch a full year's data from the API and cache it."""
        is_partial = not year_complete
        start = _year_start(year)
        end = _year_end(year)

        # For partial (current) years, fetch up to now
        if is_partial:
            end = min(end, self._now_fn())

        logger.info(
            "Fetching year %d (%s): %s to %s",
            year,
            "partial" if is_partial else "complete",
            start.isoformat(),
            end.isoformat(),
        )

        raw_df = self._crawler.fetch(
            symbol=self._bt_cfg.symbol,
            timeframe=self._bt_cfg.timeframe,
            start=start,
            end=end,
        )

        # Save raw CSV
        raw_path = self._raw_path_for_year(year, partial=is_partial)
        self._crawler.save_raw(raw_df, raw_path)

        # Process and save parquet
        processed_df = self._processor.process(raw_df)
        processed_path = self._processed_path_for_year(year, partial=is_partial)
        self._processor.save_processed(processed_df, processed_path)

        return processed_df

    def _trim_to_range(self, df: pd.DataFrame) -> pd.DataFrame:
        """Trim combined DataFrame to exact ``start_date`` / ``end_date``."""
        start = self._bt_cfg.start_date
        end = self._bt_cfg.end_date

        # Ensure timezone awareness
        if start.tzinfo is None:
            start = start.replace(tzinfo=UTC)
        if end.tzinfo is None:
            end = end.replace(tzinfo=UTC)

        mask = (df["timestamp"] >= start) & (df["timestamp"] <= end)
        return df.loc[mask].reset_index(drop=True)

    # ------------------------------------------------------------------ #
    # Path generation                                                     #
    # ------------------------------------------------------------------ #

    def _raw_path_for_year(self, year: int, *, partial: bool = False) -> Path:
        """Generate raw CSV path for a given year."""
        sym = _normalize_symbol(self._bt_cfg.symbol)
        tf = self._bt_cfg.timeframe
        suffix = "_partial" if partial else ""
        filename = f"{sym}_{tf}_{year}{suffix}.csv"
        return Path(self._data_cfg.raw_dir) / filename

    def _processed_path_for_year(self, year: int, *, partial: bool = False) -> Path:
        """Generate processed parquet path for a given year."""
        sym = _normalize_symbol(self._bt_cfg.symbol)
        tf = self._bt_cfg.timeframe
        suffix = "_partial" if partial else ""
        filename = f"{sym}_{tf}_{year}{suffix}.parquet"
        return Path(self._data_cfg.processed_dir) / filename
