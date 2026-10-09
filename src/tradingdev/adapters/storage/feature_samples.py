"""Bounded feature-cache reads for exact joins onto historical sample bars."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import pyarrow.dataset as arrow_dataset

from tradingdev.adapters.storage.parquet_samples import iter_parquet_window

if TYPE_CHECKING:
    from pathlib import Path


def read_feature_sample(
    path: Path,
    *,
    timestamps: pd.Series,
    column: str,
    max_rows: int,
    fallback_column: str | None = None,
) -> pd.DataFrame:
    """Keep only matching feature rows, rejecting excess rows before any join."""
    dataset = arrow_dataset.dataset(path, format="parquet")
    value_column = column
    if column not in dataset.schema.names and fallback_column in dataset.schema.names:
        assert fallback_column is not None
        value_column = fallback_column
    columns = ["timestamp", value_column]
    frames: list[pd.DataFrame] = []
    rows = 0
    for frame in iter_parquet_window(
        dataset,
        start=timestamps.min(),
        end=timestamps.max(),
        batch_size=max_rows,
        columns=columns,
    ):
        matched = frame.loc[frame["timestamp"].isin(timestamps)]
        if matched.empty:
            continue
        rows += len(matched)
        if rows > max_rows:
            raise ValueError("Feature joins exceeded the sample row budget")
        frames.append(matched)
    if frames:
        result = pd.concat(frames, ignore_index=True)
    else:
        result = dataset.schema.empty_table().select(columns).to_pandas()
        result["timestamp"] = pd.to_datetime(result["timestamp"], utc=True)
    return result.rename(columns={value_column: column})
