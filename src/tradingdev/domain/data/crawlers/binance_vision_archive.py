"""Bounded archive downloads and CSV sampling for Binance Vision."""

from __future__ import annotations

import zipfile
from dataclasses import dataclass
from tempfile import SpooledTemporaryFile
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from datetime import datetime

    import httpx

OHLCV_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]
_DOWNLOAD_CHUNK_BYTES = 64 * 1024
_SPOOL_MEMORY_BYTES = 256 * 1024
_ARCHIVE_DOWNLOAD_BYTES = 16 * 1024 * 1024
_SAMPLE_DOWNLOAD_BYTES = 32 * 1024 * 1024
_CSV_UNCOMPRESSED_BYTES = 128 * 1024 * 1024
_CSV_CHUNK_ROWS = 1024


def normalize_vision_rows(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalize headerless or headed Vision rows, including spot microseconds."""
    if not raw.empty and str(raw.iloc[0, 0]).strip().lower() == "open_time":
        raw = raw.iloc[1:].reset_index(drop=True)
    if raw.empty:
        return pd.DataFrame(columns=OHLCV_COLUMNS)
    raw.columns = pd.Index(range(6))
    timestamps = pd.to_numeric(raw[0])
    # Vision uses milliseconds historically and microseconds in newer spot files.
    microseconds = timestamps.where(
        timestamps.abs() >= 100_000_000_000_000, timestamps * 1000
    )
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(microseconds, unit="us", utc=True),
            "open": raw[1].astype(float),
            "high": raw[2].astype(float),
            "low": raw[3].astype(float),
            "close": raw[4].astype(float),
            "volume": raw[5].astype(float),
        }
    )


@dataclass
class SampleDownloadBudget:
    """Account for all compressed bytes consumed by one sample request."""

    downloaded: int = 0

    def check(self, archive_bytes: int) -> None:
        if archive_bytes > _ARCHIVE_DOWNLOAD_BYTES:
            raise ValueError(
                "Binance Vision sample archive exceeds download byte limit"
            )
        if self.downloaded + archive_bytes > _SAMPLE_DOWNLOAD_BYTES:
            raise ValueError("Binance Vision sample exceeds total download byte limit")


def read_sample_archive(
    client: httpx.Client,
    url: str,
    *,
    start: datetime,
    end: datetime,
    max_rows: int,
    budget: SampleDownloadBudget,
) -> pd.DataFrame | None:
    """Download one capped ZIP, then retain only the requested sample rows.

    ZIP requires a seekable archive, so the entire compressed file is downloaded.
    Spooling bounds download memory; CSV chunks bound row materialization. Each
    archive is scanned completely to select the earliest rows regardless of order.
    """
    with SpooledTemporaryFile(max_size=_SPOOL_MEMORY_BYTES, mode="w+b") as spool:
        with client.stream(
            "GET",
            url,
            headers={"Accept-Encoding": "identity"},
            follow_redirects=False,
        ) as response:
            if response.status_code == 404:
                return None
            response.raise_for_status()
            encoding = response.headers.get("content-encoding", "identity")
            if encoding.strip().lower() != "identity":
                raise ValueError(
                    "Binance Vision sample requires identity content encoding"
                )
            content_length = response.headers.get("content-length")
            if content_length is not None:
                budget.check(int(content_length))
            archive_bytes = 0
            for chunk in response.iter_raw(chunk_size=_DOWNLOAD_CHUNK_BYTES):
                archive_bytes += len(chunk)
                budget.check(archive_bytes)
                spool.write(chunk)
            budget.downloaded += archive_bytes
        spool.seek(0)
        return _read_sample_csv(spool, start=start, end=end, max_rows=max_rows)


def _read_sample_csv(
    source: SpooledTemporaryFile[bytes],
    *,
    start: datetime,
    end: datetime,
    max_rows: int,
) -> pd.DataFrame:
    selected = pd.DataFrame(columns=OHLCV_COLUMNS)
    with zipfile.ZipFile(source) as archive:
        members = [
            item for item in archive.infolist() if item.filename.endswith(".csv")
        ]
        if len(members) != 1:
            raise ValueError("Binance Vision sample archive must contain one CSV")
        member = members[0]
        if member.file_size > _CSV_UNCOMPRESSED_BYTES:
            raise ValueError(
                "Binance Vision sample CSV exceeds uncompressed byte limit"
            )
        with archive.open(member) as source_csv:
            try:
                reader = pd.read_csv(
                    source_csv,
                    header=None,
                    usecols=range(6),
                    chunksize=_CSV_CHUNK_ROWS,
                )
            except pd.errors.EmptyDataError:
                return selected
            with reader:
                for raw in reader:
                    chunk = normalize_vision_rows(raw)
                    if chunk.empty:
                        continue
                    chunk = chunk.loc[chunk["timestamp"].between(start, end)]
                    if chunk.empty:
                        continue
                    selected = (
                        (
                            chunk
                            if selected.empty
                            else pd.concat([selected, chunk], ignore_index=True)
                        )
                        .drop_duplicates("timestamp")
                        .sort_values("timestamp")
                        .head(max_rows)
                    )
    return selected.reset_index(drop=True)
