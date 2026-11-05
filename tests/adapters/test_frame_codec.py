"""Boundary serialization preserves values and rejects executable dtype metadata."""

from __future__ import annotations

import base64
import builtins
from datetime import UTC, timedelta, timezone, tzinfo
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

from tradingdev.adapters.execution.frame_codec import (
    FramePayload,
    decode_frame,
    encode_frame,
)

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Any


@pytest.mark.parametrize("index_kind", ["range", "datetime", "labels"])
def test_wire_roundtrip_preserves_precision_dtypes_missing_values_and_index(
    index_kind: str,
) -> None:
    frame = pd.DataFrame(
        {
            "float": [np.nextafter(1.0, 2.0), np.nan, -np.inf],
            "integer": pd.Series([2**62, -(2**62), 0], dtype="int64"),
            "nullable": pd.Series([None, 2**63, 7], dtype="UInt64"),
            "boolean": pd.Series([None, False, True], dtype="boolean"),
            "text": pd.Series(["中文", None, ""], dtype="str"),
            "python_text": pd.Series(
                ["中文", None, ""], dtype=pd.StringDtype(storage="python")
            ),
            "objects": pd.Series([{"nested": [1, "x"]}, None, [2]], dtype=object),
            "timestamp": pd.date_range(
                "2024-01-01", periods=3, freq="h", tz="Asia/Taipei"
            ),
        }
    )
    if index_kind == "range":
        frame.index = pd.RangeIndex(10, 4, -2, name="bars")
    elif index_kind == "datetime":
        frame.index = pd.date_range(
            "2024-02-01", periods=3, freq="h", tz="UTC", name="bars"
        )
    else:
        frame.index = pd.Index(["bar-c", "bar-a", "bar-b"], name="bars")
    frame.columns.name = "features"
    frame.attrs = {"source": "captured", "window": [1, 3]}
    serialized = encode_frame(frame).model_dump_json()
    restored = decode_frame(FramePayload.model_validate_json(serialized))
    pd.testing.assert_frame_equal(restored, frame, check_exact=True)
    assert restored.index.identical(frame.index)
    assert restored.attrs == frame.attrs


@pytest.mark.parametrize("corruption", ["dtype", "size", "range", "columns"])
def test_wire_rejects_malformed_vectors_before_constructing_arbitrary_objects(
    corruption: str,
) -> None:
    payload: dict[str, Any] = encode_frame(pd.DataFrame({"close": [1.0]})).model_dump()
    if corruption == "dtype":
        payload["vectors"][0]["dtype"] = "pickle"
        payload["vectors"][0]["data"] = base64.b64encode(b"arbitrary").decode("ascii")
    elif corruption == "size":
        payload["index"]["values"]["size"] = 10**12
    elif corruption == "range":
        payload["index"]["step"] = 0
    else:
        payload["columns"] = ["close", "close"]
    with pytest.raises(ValueError):
        decode_frame(FramePayload.model_validate(payload))


def test_unsupported_python_objects_are_rejected_without_coercion() -> None:
    class Arbitrary:
        def __str__(self) -> str:
            pytest.fail("Python objects must not be coerced through user hooks")

    with pytest.raises(ValueError, match="finite JSON values"):
        encode_frame(pd.DataFrame({"object": [Arbitrary()]}))


@pytest.mark.parametrize(
    "zone",
    [
        UTC,
        "America/New_York",
        timezone(timedelta(hours=5, minutes=30)),
        timezone(-timedelta(hours=3, seconds=15, microseconds=123456)),
    ],
)
def test_timezone_roundtrip_uses_safe_names_and_fixed_offsets(
    zone: str | tzinfo,
) -> None:
    index = pd.date_range("2024-03-10", periods=8, freq="h", tz=zone, name="bars")
    frame = pd.DataFrame({"timestamp": index}, index=index)
    restored = decode_frame(
        FramePayload.model_validate_json(encode_frame(frame).model_dump_json())
    )
    pd.testing.assert_frame_equal(restored, frame, check_exact=True)
    assert restored.index.identical(index)


@pytest.mark.parametrize("path_kind", ["dateutil", "absolute", "traversal"])
def test_worker_timezone_cannot_open_a_host_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, path_kind: str
) -> None:
    sentinel = tmp_path / "sentinel"
    sentinel.write_bytes(b"not a timezone")
    payload = encode_frame(
        pd.DataFrame({"timestamp": pd.date_range("2024-01-01", periods=1, tz="UTC")})
    ).model_dump()
    payload["vectors"][0]["timezone"] = {
        "dateutil": "dateutil/" + str(sentinel),
        "absolute": str(sentinel),
        "traversal": "../" + str(sentinel).lstrip("/"),
    }[path_kind]
    opened: list[object] = []
    original = builtins.open

    def observed(file: Any, *args: Any, **kwargs: Any) -> Any:
        if str(file) == str(sentinel):
            opened.append(file)
        return original(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", observed)
    with pytest.raises(ValueError, match="Unsupported isolated frame timezone"):
        decode_frame(FramePayload.model_validate(payload))
    assert opened == []
