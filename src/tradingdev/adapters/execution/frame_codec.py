"""A closed, lossless DataFrame wire format without executable serialization.

Numeric buffers have explicit primitive dtypes and exact byte lengths. Text and
object values are plain JSON. No pickle, extension class, compression, filesystem
path, or user-provided dtype is passed to a deserializer.
"""

from __future__ import annotations

import base64
import re
from datetime import UTC, timedelta, timezone, tzinfo
from typing import Any, Literal, cast
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, JsonValue

from tradingdev.domain.strategies.execution import finite_strategy_value

_NUMERIC = {
    name: np.dtype(name)
    for name in (
        "bool",
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "float32",
        "float64",
    )
}
_NULLABLE = {
    name: name.lower()
    for name in (
        "Int8",
        "Int16",
        "Int32",
        "Int64",
        "UInt8",
        "UInt16",
        "UInt32",
        "UInt64",
        "Float32",
        "Float64",
    )
}
_NULLABLE["boolean"] = "bool"
_UNITS = {"s", "ms", "us", "ns"}


def _timezone(value: str) -> tzinfo:
    """Never pass a worker string to pandas' path-aware dateutil resolver."""
    if value == "UTC":
        return UTC
    offset = re.fullmatch(
        r"UTC([+-])([0-9]{2}):([0-9]{2})(?::([0-9]{2})(?:\.([0-9]{1,6}))?)?",
        value,
    )
    if offset is not None:
        sign, hours, minutes, seconds, fraction = offset.groups()
        if int(hours) > 23 or int(minutes) > 59 or int(seconds or "0") > 59:
            raise ValueError("Unsupported isolated frame timezone offset")
        delta = timedelta(
            hours=int(hours),
            minutes=int(minutes),
            seconds=int(seconds or "0"),
            microseconds=int((fraction or "").ljust(6, "0")),
        )
        return timezone(-delta if sign == "-" else delta)
    if re.fullmatch(r"[A-Za-z0-9_+-]+(?:/[A-Za-z0-9_+-]+)*", value) is not None:
        try:
            # ZoneInfo constrains keys to the trusted timezone search path. It
            # does not interpret dateutil prefixes or arbitrary absolute paths.
            return ZoneInfo(value)
        except ZoneInfoNotFoundError as error:
            raise ValueError("Unsupported isolated frame timezone") from error
    raise ValueError("Unsupported isolated frame timezone")


class Vector(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    dtype: str = Field(max_length=64)
    size: int = Field(ge=0, strict=True)
    data: str = ""
    mask: str | None = None
    values: list[JsonValue] | None = None
    unit: Literal["s", "ms", "us", "ns"] | None = None
    timezone: str | None = Field(default=None, max_length=128)
    storage: Literal["python", "pyarrow"] | None = None


class FrameIndex(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    kind: Literal["range", "datetime", "values"]
    name: str | None = None
    values: Vector
    start: int = 0
    step: int = 1
    frequency: str | None = Field(default=None, max_length=128)


class FramePayload(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    index: FrameIndex
    columns: list[str]
    column_name: str | None = None
    vectors: list[Vector]
    attrs: dict[str, JsonValue] = Field(default_factory=dict)


def _buffer(values: np.ndarray[Any, Any]) -> str:
    return base64.b64encode(values.tobytes()).decode("ascii")


def _array(data: str, dtype: np.dtype[Any], size: int) -> np.ndarray[Any, Any]:
    decoded = base64.b64decode(data, validate=True)
    if len(decoded) != size * dtype.itemsize:
        raise ValueError("Frame buffer length does not match its declared vector")
    return np.frombuffer(decoded, dtype=dtype).copy()


def _encode_vector(series: pd.Series[Any]) -> Vector:
    dtype = str(series.dtype)
    size = len(series)
    if dtype in _NUMERIC:
        return Vector(dtype=dtype, size=size, data=_buffer(series.to_numpy()))
    if dtype in _NULLABLE:
        primitive = _NUMERIC[_NULLABLE[dtype]]
        return Vector(
            dtype=dtype,
            size=size,
            data=_buffer(series.to_numpy(dtype=primitive, na_value=0)),
            mask=_buffer(series.isna().to_numpy()),
        )
    if isinstance(series.dtype, pd.DatetimeTZDtype) or dtype.startswith("datetime64["):
        unit = series.dt.unit
        if unit not in _UNITS:
            raise ValueError("Unsupported datetime unit")
        zone = str(series.dt.tz) if series.dt.tz is not None else None
        if zone is not None:
            _timezone(zone)
        return Vector(
            dtype="datetime",
            size=size,
            data=_buffer(series.array.asi8),
            unit=cast("Literal['s', 'ms', 'us', 'ns']", unit),
            timezone=zone,
        )
    if isinstance(series.dtype, pd.StringDtype):
        return Vector(
            dtype="str" if series.dtype.na_value is np.nan else "string",
            size=size,
            values=[None if pd.isna(value) else str(value) for value in series],
            storage=series.dtype.storage,
        )
    if dtype == "object":
        return Vector(
            dtype=dtype,
            size=size,
            values=[finite_strategy_value(value) for value in series],
        )
    raise ValueError(f"Unsupported isolated frame dtype: {dtype}")


def _decode_vector(vector: Vector) -> pd.Series[Any]:
    dtype = vector.dtype
    if dtype in _NUMERIC or dtype in _NULLABLE:
        primitive = _NUMERIC[_NULLABLE.get(dtype, dtype)]
        values = _array(vector.data, primitive, vector.size)
        result = pd.Series(values, dtype=dtype)
        if dtype in _NULLABLE:
            if vector.mask is None:
                raise ValueError("A nullable vector requires its missing-value mask")
            result[_array(vector.mask, _NUMERIC["bool"], vector.size)] = pd.NA
        return result
    if dtype == "datetime":
        if vector.unit is None:
            raise ValueError("A datetime vector requires its unit")
        values = _array(vector.data, _NUMERIC["int64"], vector.size)
        timestamps = pd.DatetimeIndex(values.view(f"datetime64[{vector.unit}]"))
        if vector.timezone is not None:
            timestamps = timestamps.tz_localize(UTC).tz_convert(
                _timezone(vector.timezone)
            )
        return pd.Series(timestamps)
    if dtype not in {"str", "string", "object"}:
        raise ValueError(f"Unsupported isolated frame dtype: {dtype}")
    if vector.values is None or len(vector.values) != vector.size:
        raise ValueError("Frame values do not match their declared vector")
    if dtype != "object" and any(
        value is not None and not isinstance(value, str) for value in vector.values
    ):
        raise ValueError("A string vector contains non-string values")
    finite_strategy_value(vector.values)
    if dtype == "object":
        return pd.Series(vector.values, dtype=object)
    if vector.storage is None:
        raise ValueError("A string vector requires its storage kind")
    return pd.Series(
        vector.values,
        dtype=pd.StringDtype(
            storage=vector.storage,
            na_value=np.nan if dtype == "str" else pd.NA,
        ),
    )


def encode_frame(frame: pd.DataFrame) -> FramePayload:
    if (
        not frame.columns.is_unique
        or not all(isinstance(column, str) for column in frame.columns)
        or not isinstance(frame.index.name, str | type(None))
        or not isinstance(frame.columns.name, str | type(None))
    ):
        raise ValueError(
            "Isolated frames require unique string columns and string names"
        )
    index = frame.index
    if isinstance(index, pd.MultiIndex):
        raise ValueError("Isolated frames do not support a MultiIndex")
    kind: Literal["range", "datetime", "values"] = (
        "range"
        if isinstance(index, pd.RangeIndex)
        else "datetime"
        if isinstance(index, pd.DatetimeIndex)
        else "values"
    )
    attrs = finite_strategy_value(frame.attrs)
    assert isinstance(attrs, dict)
    return FramePayload(
        index=FrameIndex(
            kind=kind,
            name=index.name,
            values=_encode_vector(pd.Series(index)),
            start=index.start if isinstance(index, pd.RangeIndex) else 0,
            step=index.step if isinstance(index, pd.RangeIndex) else 1,
            frequency=index.freqstr if isinstance(index, pd.DatetimeIndex) else None,
        ),
        columns=list(frame.columns),
        column_name=frame.columns.name,
        vectors=[
            _encode_vector(frame.iloc[:, index]) for index in range(len(frame.columns))
        ],
        attrs=attrs,
    )


def decode_frame(payload: FramePayload) -> pd.DataFrame:
    if len(set(payload.columns)) != len(payload.columns) or len(payload.columns) != len(
        payload.vectors
    ):
        raise ValueError("Frame column identities do not match their vectors")
    index_values = _decode_vector(payload.index.values)
    size = len(index_values)
    if any(vector.size != size for vector in payload.vectors):
        raise ValueError("Frame vectors have inconsistent row counts")
    index: pd.Index[Any]
    if payload.index.kind == "range":
        if payload.index.step == 0:
            raise ValueError("Range index step cannot be zero")
        index = pd.RangeIndex(
            payload.index.start,
            payload.index.start + size * payload.index.step,
            payload.index.step,
            name=payload.index.name,
        )
        if not np.array_equal(index.to_numpy(), index_values.to_numpy()):
            raise ValueError("Range index values do not match its geometry")
    elif payload.index.kind == "datetime":
        if payload.index.values.dtype != "datetime":
            raise ValueError("Datetime index requires datetime values")
        index = (
            pd.DatetimeIndex(
                index_values, name=payload.index.name, freq=payload.index.frequency
            )
            if payload.index.frequency is not None
            else pd.DatetimeIndex(index_values, name=payload.index.name)
        )
    else:
        index = pd.Index(index_values, name=payload.index.name)
    frame = pd.DataFrame(
        {
            name: _decode_vector(vector).array
            for name, vector in zip(payload.columns, payload.vectors, strict=True)
        },
        index=index,
    )
    frame.columns.name = payload.column_name
    frame.attrs.update(payload.attrs)
    return frame
