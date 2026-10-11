"""Vision samples cap archive costs and close each streaming resource offline."""

from __future__ import annotations

import io
import zipfile
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta, timezone
from tempfile import SpooledTemporaryFile
from typing import TYPE_CHECKING, Any

import httpx
import pandas as pd
import pytest

from tradingdev.domain.data.crawlers import binance_vision_archive as archives
from tradingdev.domain.data.crawlers.binance_vision import (
    BinanceVisionCrawler,
    _parse_zip_csv,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from pathlib import Path

_START = datetime(2024, 1, 1, tzinfo=UTC)


def _zip(timestamps: list[int], *, header: bool = False) -> bytes:
    lines = [f"{value},1,2,0.5,1.5,3,0,0,0,0,0,0" for value in timestamps]
    if header:
        lines.insert(0, "open_time,open,high,low,close,volume,unused")
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("sample.csv", "\n".join(lines))
    return output.getvalue()


def _times(count: int, *, start: datetime = _START, unit: str = "ms") -> list[int]:
    scale = 1000 if unit == "ms" else 1_000_000
    return [
        int((start + timedelta(minutes=index)).timestamp() * scale)
        for index in range(count)
    ]


class _Stream(httpx.SyncByteStream):
    def __init__(self, content: bytes, error: Exception | None = None) -> None:
        self.content = content
        self.error = error
        self.closed = False

    def __iter__(self) -> Iterator[bytes]:
        yield self.content
        if self.error is not None:
            raise self.error

    def close(self) -> None:
        self.closed = True


@contextmanager
def _crawler(
    reply: Callable[[httpx.Request], httpx.Response],
) -> Iterator[BinanceVisionCrawler]:
    crawler = BinanceVisionCrawler()
    crawler._client.close()
    crawler._client = httpx.Client(transport=httpx.MockTransport(reply))
    try:
        yield crawler
    finally:
        crawler._client.close()


@pytest.mark.parametrize("header", [False, True])
@pytest.mark.parametrize("unit", ["ms", "us"])
def test_sample_limits_csv_batches_and_stops_after_first_daily_archive(
    monkeypatch: pytest.MonkeyPatch, header: bool, unit: str
) -> None:
    payload = _zip(_times(10000, unit=unit), header=header)
    requested: list[str] = []
    batches: list[int] = []
    normalize = archives.normalize_vision_rows

    def record_rows(frame: pd.DataFrame) -> pd.DataFrame:
        batches.append(len(frame))
        return normalize(frame)

    def reply(request: httpx.Request) -> httpx.Response:
        requested.append(request.url.path)
        return httpx.Response(200, stream=_Stream(payload))

    monkeypatch.setattr(archives, "normalize_vision_rows", record_rows)
    with _crawler(reply) as crawler:
        result = crawler.fetch_sample(
            "BTC/USDT", "1m", _START, _START + timedelta(days=30), max_rows=64
        )
    assert len(requested) == 1
    assert "/daily/" in requested[0] and requested[0].endswith("2024-01-01.zip")
    assert len(result) == 64
    assert result["timestamp"].iloc[0] == pd.Timestamp(_START)
    assert result["timestamp"].iloc[-1] == pd.Timestamp(_START + timedelta(minutes=63))
    assert all(rows <= 1024 for rows in batches)
    assert sum(batches) == 10000 + int(header)


@pytest.mark.parametrize("header", [False, True])
def test_sample_keeps_earliest_unique_rows_across_unordered_csv_batches(
    monkeypatch: pytest.MonkeyPatch,
    header: bool,
) -> None:
    monkeypatch.setattr(archives, "_CSV_CHUNK_ROWS", 2)
    values = _times(4)
    payload = _zip([values[index] for index in [2, 3, 2, 0, 1, 0]], header=header)
    with _crawler(
        lambda _request: httpx.Response(200, stream=_Stream(payload))
    ) as crawler:
        result = crawler.fetch_sample(
            "BTC/USDT", "1m", _START, _START + timedelta(minutes=3), max_rows=2
        )
    assert result["timestamp"].tolist() == [
        pd.Timestamp(_START),
        pd.Timestamp(_START + timedelta(minutes=1)),
    ]


@pytest.mark.parametrize("header", [False, True])
def test_ordinary_parser_and_sample_share_microsecond_support(header: bool) -> None:
    result = _parse_zip_csv(_zip(_times(2, unit="us"), header=header))
    assert result["timestamp"].tolist() == [
        pd.Timestamp(_START),
        pd.Timestamp(_START + timedelta(minutes=1)),
    ]


def test_sample_handles_duplicate_rows_and_inclusive_single_timestamp() -> None:
    values = _times(4)
    payload = _zip([values[0], values[0], values[1], values[1], values[2], values[3]])
    with _crawler(
        lambda _request: httpx.Response(200, stream=_Stream(payload))
    ) as crawler:
        result = crawler.fetch_sample(
            "BTC/USDT", "1m", _START, _START + timedelta(minutes=3), max_rows=3
        )
        exact = crawler.fetch_sample(
            "BTC/USDT",
            "1m",
            _START.replace(tzinfo=None),
            _START.replace(tzinfo=None),
            max_rows=1,
        )
    assert result["timestamp"].tolist() == [
        pd.Timestamp(_START + timedelta(minutes=i)) for i in range(3)
    ]
    assert exact["timestamp"].tolist() == [pd.Timestamp(_START)]


def test_monthly_fallback_runs_once_and_skips_covered_days() -> None:
    requested: list[str] = []
    payload = _zip(_times(2, start=datetime(2024, 1, 31, tzinfo=UTC)))

    def reply(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        requested.append(path)
        if "/monthly/" in path and path.endswith("2024-01.zip"):
            return httpx.Response(200, stream=_Stream(payload))
        if path.endswith("2024-02-01.zip"):
            return httpx.Response(
                200,
                stream=_Stream(_zip(_times(2, start=datetime(2024, 2, 1, tzinfo=UTC)))),
            )
        return httpx.Response(404)

    with _crawler(reply) as crawler:
        result = crawler.fetch_sample(
            "BTC/USDT", "1m", _START, datetime(2024, 2, 2, tzinfo=UTC), max_rows=3
        )
    assert len(result) == 3
    assert [path.rsplit("/", 1)[-1] for path in requested] == [
        "BTCUSDT-1m-2024-01-01.zip",
        "BTCUSDT-1m-2024-01.zip",
        "BTCUSDT-1m-2024-02-01.zip",
    ]


def test_missing_monthly_archive_is_not_retried_for_each_missing_day() -> None:
    requested: list[str] = []

    def reply(request: httpx.Request) -> httpx.Response:
        requested.append(request.url.path)
        return httpx.Response(404)

    with _crawler(reply) as crawler:
        result = crawler.fetch_sample(
            "BTC/USDT", "1m", _START, _START + timedelta(days=2), max_rows=64
        )
    assert result.empty
    assert sum("/monthly/" in path for path in requested) == 1
    assert sum("/daily/" in path for path in requested) == 3


@pytest.mark.parametrize("status", [403, 500])
def test_http_failure_does_not_fall_back(status: int) -> None:
    requested: list[str] = []

    def reply(request: httpx.Request) -> httpx.Response:
        requested.append(request.url.path)
        return httpx.Response(status)

    with _crawler(reply) as crawler, pytest.raises(httpx.HTTPStatusError):
        crawler.fetch_sample("BTC/USDT", "1m", _START, _START, max_rows=1)
    assert len(requested) == 1


@pytest.mark.parametrize("content_length", [None, "50", "500"])
def test_archive_download_cap_rejects_declared_and_actual_stream_bytes(
    monkeypatch: pytest.MonkeyPatch, content_length: str | None
) -> None:
    monkeypatch.setattr(archives, "_ARCHIVE_DOWNLOAD_BYTES", 100)
    monkeypatch.setattr(archives, "_DOWNLOAD_CHUNK_BYTES", 16)
    headers = {} if content_length is None else {"content-length": content_length}
    stream = _Stream(b"x" * 200)
    with (
        _crawler(
            lambda _request: httpx.Response(200, headers=headers, stream=stream)
        ) as crawler,
        pytest.raises(ValueError, match="archive exceeds download byte limit"),
    ):
        crawler.fetch_sample("BTC/USDT", "1m", _START, _START, max_rows=1)
    assert stream.closed


def test_total_download_cap_includes_previous_archives(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _zip(_times(1))
    second = _zip(_times(1, start=_START + timedelta(days=1)))
    monkeypatch.setattr(
        archives, "_SAMPLE_DOWNLOAD_BYTES", len(first) + len(second) - 1
    )
    streams = [_Stream(first), _Stream(second)]
    replies = iter(streams)
    with (
        _crawler(lambda _request: httpx.Response(200, stream=next(replies))) as crawler,
        pytest.raises(ValueError, match="total download byte limit"),
    ):
        crawler.fetch_sample(
            "BTC/USDT", "1m", _START, _START + timedelta(days=1), max_rows=2
        )
    assert all(stream.closed for stream in streams)


def test_archive_uncompressed_size_is_checked_before_csv_parsing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(archives, "_CSV_UNCOMPRESSED_BYTES", 10)
    payload = _zip(_times(4))

    def unexpected(*_args: Any, **_kwargs: Any) -> None:
        pytest.fail("Oversized archive must be rejected before CSV parsing")

    monkeypatch.setattr(pd, "read_csv", unexpected)
    with (
        _crawler(
            lambda _request: httpx.Response(200, stream=_Stream(payload))
        ) as crawler,
        pytest.raises(ValueError, match="uncompressed byte limit"),
    ):
        crawler.fetch_sample("BTC/USDT", "1m", _START, _START, max_rows=1)


@pytest.mark.parametrize(
    "outcome", ["success", "download_failure", "timeout", "csv_failure"]
)
def test_sampling_closes_response_spool_and_zip_without_temporary_file_leaks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, outcome: str
) -> None:
    spools: list[Any] = []
    opened_archives: list[zipfile.ZipFile] = []
    real_zip = zipfile.ZipFile
    payload = _zip(_times(3))
    error = (
        httpx.ReadTimeout("injected timeout")
        if outcome == "timeout"
        else httpx.ReadError("injected download failure")
        if outcome == "download_failure"
        else None
    )
    stream = _Stream(payload, error=error)

    def spool(**kwargs: Any) -> Any:
        kwargs["max_size"] = 1
        # The production caller must enter and close this tracked context.
        result = SpooledTemporaryFile(dir=str(tmp_path), **kwargs)  # noqa: SIM115
        spools.append(result)
        return result

    def open_zip(*args: Any, **kwargs: Any) -> zipfile.ZipFile:
        result = real_zip(*args, **kwargs)
        opened_archives.append(result)
        return result

    def fail_csv(_frame: pd.DataFrame) -> pd.DataFrame:
        raise ValueError("injected CSV failure")

    monkeypatch.setattr(archives, "SpooledTemporaryFile", spool)
    monkeypatch.setattr(archives, "_DOWNLOAD_CHUNK_BYTES", 16)
    monkeypatch.setattr(zipfile, "ZipFile", open_zip)
    if outcome == "csv_failure":
        monkeypatch.setattr(archives, "normalize_vision_rows", fail_csv)
    with _crawler(lambda _request: httpx.Response(200, stream=stream)) as crawler:
        if outcome == "success":
            assert (
                len(crawler.fetch_sample("BTC/USDT", "1m", _START, _START, max_rows=1))
                == 1
            )
        else:
            with pytest.raises((httpx.RequestError, ValueError), match="injected"):
                crawler.fetch_sample("BTC/USDT", "1m", _START, _START, max_rows=1)
    assert stream.closed
    assert spools and all(handle.closed for handle in spools)
    assert all(handle._rolled for handle in spools)
    assert all(archive.fp is None for archive in opened_archives)
    assert len(opened_archives) == (1 if outcome in {"success", "csv_failure"} else 0)
    assert list(tmp_path.iterdir()) == []


def test_sample_uses_utc_dates_and_limits_fetch_to_inclusive_end() -> None:
    local = timezone(timedelta(hours=8))
    start = datetime(2024, 1, 2, 7, 59, tzinfo=local)
    end = datetime(2024, 1, 2, 8, 0, tzinfo=local)
    requested: list[str] = []
    timestamps = [
        int(datetime(2024, 1, 1, 23, 59, tzinfo=UTC).timestamp() * 1000),
        int(datetime(2024, 1, 2, tzinfo=UTC).timestamp() * 1000),
    ]

    def reply(request: httpx.Request) -> httpx.Response:
        requested.append(request.url.path)
        # Both archive bounds and the requested UTC bounds must be respected.
        return httpx.Response(200, stream=_Stream(_zip(timestamps)))

    with _crawler(reply) as crawler:
        result = crawler.fetch_sample("BTC/USDT", "1m", start, end, max_rows=64)
    assert [path.rsplit("/", 1)[-1] for path in requested] == [
        "BTCUSDT-1m-2024-01-01.zip",
        "BTCUSDT-1m-2024-01-02.zip",
    ]
    assert result["timestamp"].tolist() == [pd.Timestamp(start), pd.Timestamp(end)]


@pytest.mark.parametrize("empty_csv", [False, True])
def test_empty_and_header_only_archives_return_empty_samples(empty_csv: bool) -> None:
    payload = _zip([], header=not empty_csv)
    with _crawler(
        lambda _request: httpx.Response(200, stream=_Stream(payload))
    ) as crawler:
        result = crawler.fetch_sample("BTC/USDT", "1m", _START, _START, max_rows=1)
    assert result.empty


def test_sample_rejects_nonpositive_budget_before_downloading() -> None:
    def unexpected(_request: httpx.Request) -> httpx.Response:
        pytest.fail("Invalid sample limits must not download archives")

    with _crawler(unexpected) as crawler, pytest.raises(ValueError, match="positive"):
        crawler.fetch_sample("BTC/USDT", "1m", _START, _START, max_rows=0)


def test_sample_rejects_http_content_encoding_before_consuming_body() -> None:
    class UnreadableStream(_Stream):
        def __iter__(self) -> Iterator[bytes]:
            pytest.fail("HTTP-encoded archive body must not be consumed")

    stream = UnreadableStream(b"")

    def reply(request: httpx.Request) -> httpx.Response:
        assert request.headers["accept-encoding"] == "identity"
        return httpx.Response(200, headers={"content-encoding": "gzip"}, stream=stream)

    with (
        _crawler(reply) as crawler,
        pytest.raises(ValueError, match="requires identity content encoding"),
    ):
        crawler.fetch_sample("BTC/USDT", "1m", _START, _START, max_rows=1)
    assert stream.closed


@pytest.mark.parametrize("status", [302, 307])
@pytest.mark.parametrize("encoding", ["identity", "gzip"])
def test_sample_rejects_redirect_without_reading_or_following_its_body(
    status: int, encoding: str
) -> None:
    class UnreadableStream(_Stream):
        def __iter__(self) -> Iterator[bytes]:
            pytest.fail("Redirect response body must not be consumed")

    stream = UnreadableStream(b"")
    requested: list[str] = []

    def reply(request: httpx.Request) -> httpx.Response:
        requested.append(str(request.url))
        return httpx.Response(
            status,
            headers={
                "location": "https://data.binance.vision/redirected.zip",
                "content-encoding": encoding,
                "content-length": str(archives._ARCHIVE_DOWNLOAD_BYTES + 1),
            },
            stream=stream,
        )

    with _crawler(reply) as crawler, pytest.raises(httpx.HTTPStatusError):
        crawler._client.follow_redirects = True
        crawler.fetch_sample("BTC/USDT", "1m", _START, _START, max_rows=1)
    assert len(requested) == 1
    assert stream.closed
