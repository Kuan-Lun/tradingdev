"""Bounded Parquet reads with the same UTC interpretation as ordinary loading."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import pyarrow.dataset as arrow_dataset

from tradingdev.adapters.storage.parquet_samples import iter_parquet_window

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
    selected = pd.DataFrame()
    for frame in iter_parquet_window(
        dataset,
        start=start,
        end=end,
        batch_size=max_rows,
    ):
        if frame.empty:
            continue
        selected = pd.concat([selected, frame], ignore_index=True)
        selected = (
            selected.sort_values("timestamp")
            .drop_duplicates("timestamp")
            .head(max_rows)
        )
    return selected
