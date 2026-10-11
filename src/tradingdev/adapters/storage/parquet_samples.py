"""Read projected Parquet windows in bounded batches with UTC timestamps."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import pyarrow as arrow
import pyarrow.dataset as arrow_dataset
from pyarrow import types as arrow_types

if TYPE_CHECKING:
    from collections.abc import Iterator
    from datetime import datetime


def iter_parquet_window(
    dataset: arrow_dataset.Dataset,
    *,
    start: datetime,
    end: datetime,
    batch_size: int,
    columns: list[str] | None = None,
) -> Iterator[pd.DataFrame]:
    """Normalize batches as ordinary loading does, including nonnative timestamps."""
    timestamp_type = dataset.schema.field("timestamp").type
    predicate = None
    if arrow_types.is_timestamp(timestamp_type):
        lower, upper = pd.Timestamp(start), pd.Timestamp(end)
        if timestamp_type.tz is None:
            lower, upper = lower.tz_localize(None), upper.tz_localize(None)
        else:
            lower = lower.tz_convert(timestamp_type.tz)
            upper = upper.tz_convert(timestamp_type.tz)
        timestamp = arrow_dataset.field("timestamp")
        # Explicit scalar types preserve nanoseconds that datetime conversion loses.
        predicate = (timestamp >= arrow.scalar(lower, type=timestamp_type)) & (
            timestamp <= arrow.scalar(upper, type=timestamp_type)
        )
    scanner = dataset.scanner(
        columns=columns,
        filter=predicate,
        batch_size=batch_size,
        batch_readahead=0,
        fragment_readahead=0,
    )
    for batch in scanner.to_batches():
        frame = batch.to_pandas()
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        yield frame.loc[frame["timestamp"].between(start, end)]
