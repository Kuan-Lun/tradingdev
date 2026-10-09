"""Bounded Parquet reads with the same UTC interpretation as ordinary loading."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import pyarrow.dataset as arrow_dataset
from pyarrow import types as arrow_types

if TYPE_CHECKING:
    from datetime import datetime
    from pathlib import Path


def read_market_sample(
    path: Path, *, start: datetime, end: datetime, max_rows: int
) -> pd.DataFrame:
    """Retain the first distinct UTC bars in range without loading a whole year.

    Native timestamps support predicate pushdown with matching timezone bounds.
    Other stored types follow the ordinary loader's pandas conversion in batches.
    Each batch and the retained result are bounded even for unsorted caches.
    """
    dataset = arrow_dataset.dataset(path, format="parquet")
    timestamp_type = dataset.schema.field("timestamp").type
    predicate = None
    if arrow_types.is_timestamp(timestamp_type):
        if timestamp_type.tz is None:
            lower, upper = start.replace(tzinfo=None), end.replace(tzinfo=None)
        else:
            lower = pd.Timestamp(start).tz_convert(timestamp_type.tz).to_pydatetime()
            upper = pd.Timestamp(end).tz_convert(timestamp_type.tz).to_pydatetime()
        timestamp = arrow_dataset.field("timestamp")
        predicate = (timestamp >= lower) & (timestamp <= upper)
    selected = pd.DataFrame()
    scanner = dataset.scanner(
        filter=predicate,
        batch_size=max_rows,
        batch_readahead=0,
        fragment_readahead=0,
    )
    for batch in scanner.to_batches():
        frame = batch.to_pandas()
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        frame = frame.loc[frame["timestamp"].between(start, end)]
        if frame.empty:
            continue
        selected = pd.concat([selected, frame], ignore_index=True)
        selected = (
            selected.sort_values("timestamp")
            .drop_duplicates("timestamp")
            .head(max_rows)
        )
    return selected
