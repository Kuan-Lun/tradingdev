"""Find historical samples across market closures using bounded fetch windows."""

from __future__ import annotations

import re
from datetime import UTC, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from tradingdev.adapters.storage.market_samples import read_market_sample
from tradingdev.domain.data.data_manager import market_data_filename
from tradingdev.domain.data.processor import DataProcessor

if TYPE_CHECKING:
    from tradingdev.domain.data.crawlers.base import BaseCrawler
    from tradingdev.domain.data.schemas import DataConfig, MarketDataRequest


def load_market_sample(
    request: MarketDataRequest,
    config: DataConfig,
    crawler: BaseCrawler,
    *,
    max_rows: int,
) -> pd.DataFrame:
    """Search only the requested interval; the supervising worker owns the deadline."""
    match = re.fullmatch(r"(\d+)(s|m|h|d|w|wk|mo|M)", request.timeframe)
    if match is None:
        raise ValueError(f"Unsupported sample timeframe: {request.timeframe}")
    units = {
        "s": 1,
        "m": 60,
        "h": 3600,
        "d": 86400,
        "w": 604800,
        "wk": 604800,
        "mo": 2678400,
        "M": 2678400,
    }
    seconds = int(match[1]) * units[match[2]]
    if seconds <= 0:
        raise ValueError("Sample timeframe must be positive")
    start = request.start_date
    start = start.replace(tzinfo=UTC) if start.tzinfo is None else start.astimezone(UTC)
    end = request.end_date
    end = end.replace(tzinfo=UTC) if end.tzinfo is None else end.astimezone(UTC)
    selected = pd.DataFrame()
    cursor = start
    width = seconds * max_rows * 3
    while cursor <= end:
        # Compare the requested range before constructing a timedelta or adding it.
        stop = (
            end
            if width >= (end - cursor).total_seconds()
            else cursor + timedelta(seconds=width)
        )
        fetched: pd.DataFrame | None = None
        for year in range(cursor.year, stop.year + 1):
            complete = False
            for partial in (False, True):
                path = Path(config.processed_dir) / market_data_filename(
                    request.symbol, request.timeframe, year, partial=partial
                )
                if path.exists():
                    frame = read_market_sample(
                        path, start=cursor, end=stop, max_rows=max_rows
                    )
                    selected = _accumulate(selected, frame, max_rows)
                    complete = not partial
                    break
            if len(selected) >= max_rows:
                break
            if not complete:
                if fetched is None:
                    # Fetch a contiguous window once to preserve midnight ends,
                    # but consume each year only after checking its own cache.
                    raw = crawler.fetch(request.symbol, request.timeframe, cursor, stop)
                    fetched = pd.DataFrame()
                    if not raw.empty:
                        raw = raw.copy()
                        raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True)
                        fetched = DataProcessor().process(raw)
                        fetched = fetched.loc[
                            fetched["timestamp"].between(cursor, stop)
                        ]
                if not fetched.empty:
                    frame = fetched.loc[fetched["timestamp"].dt.year == year]
                    selected = _accumulate(selected, frame, max_rows)
            if len(selected) >= max_rows:
                break
        if len(selected) >= max_rows or stop == end:
            break
        # Overlap inclusive boundaries to preserve candles; deduplicate above.
        cursor = stop
    if selected.empty:
        raise ValueError(
            "No historical bars are available in the requested sample interval"
        )
    return selected.reset_index(drop=True)


def _accumulate(
    selected: pd.DataFrame, frame: pd.DataFrame, max_rows: int
) -> pd.DataFrame:
    if frame.empty:
        return selected
    return (
        pd.concat([selected, frame], ignore_index=True)
        .drop_duplicates("timestamp")
        .sort_values("timestamp")
        .head(max_rows)
    )
