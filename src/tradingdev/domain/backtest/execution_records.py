"""Typed observations of native VectorBT orders and generic account balances.

Order attempts are distinct from paired trades: a single reversal order can
close one trade and open another. Native logs supply cash/position transitions;
their ``value``/``new_value`` are deliberately unused because default VectorBT
simulation does not update those values after fills. Instead, both sides of an
attempt are marked at its explicitly recorded, pre-slippage request price.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Literal, Self

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

if TYPE_CHECKING:
    import vectorbt as vbt


class ExecutionModel(BaseModel):
    """Finite, non-coercing JSON records with immutable scalar snapshots."""

    model_config = ConfigDict(
        extra="forbid", strict=True, frozen=True, allow_inf_nan=False
    )


class MarketReference(ExecutionModel):
    """Original bar OHLC, not an intrabar quote or an order-book observation."""

    open: float | None = Field(default=None, gt=0)
    high: float | None = Field(default=None, gt=0)
    low: float | None = Field(default=None, gt=0)
    close: float = Field(gt=0)


class AccountSnapshot(ExecutionModel):
    """Native generic balances plus equity at the enclosing valuation price."""

    cash: float
    position: float
    free_cash: float
    debt: float = Field(ge=0)
    equity: float


class ExecutionRecord(ExecutionModel):
    """One logged order attempt, cross-checked against orders when filled."""

    execution_id: int = Field(ge=0)
    order_id: int | None = Field(default=None, ge=0)
    bar_index: int = Field(ge=0)
    timestamp: str | None
    status: Literal["filled", "ignored", "rejected"]
    status_info: str | None
    side: Literal["buy", "sell"] | None
    requested_size: float | None
    requested_size_kind: Literal[
        "finite", "positive_infinity", "negative_infinity", "not_a_number"
    ]
    requested_size_type: str
    requested_direction: str
    requested_price: float = Field(gt=0)
    requested_fee_rate: float
    requested_fixed_fees: float
    requested_slippage: float
    filled_size: float | None = Field(default=None, gt=0)
    filled_price: float | None = Field(default=None, gt=0)
    fees: float | None = None
    market: MarketReference
    valuation_price: float = Field(gt=0)
    before: AccountSnapshot
    after: AccountSnapshot

    @model_validator(mode="after")
    def coherent_execution(self) -> Self:
        """Never label a rejected attempt as a fill, or emit stale log equity."""
        fill_values = (
            self.order_id,
            self.side,
            self.filled_size,
            self.filled_price,
            self.fees,
        )
        if self.status == "filled":
            if any(value is None for value in fill_values):
                raise ValueError("Filled executions require complete order details")
        elif any(value is not None for value in fill_values):
            raise ValueError("Unfilled executions cannot contain order details")
        if (self.requested_size_kind == "finite") != (self.requested_size is not None):
            raise ValueError("Requested size and its kind disagree")
        # Both fields copy the same native request price, without arithmetic.
        if self.valuation_price != self.requested_price:
            raise ValueError("Execution valuation price differs from requested price")
        for state in (self.before, self.after):
            _check_equal(
                state.equity,
                state.cash + state.position * self.valuation_price,
                "Execution equity differs from cash plus marked position",
            )
        if self.status == "filled":
            assert self.filled_size is not None
            assert self.filled_price is not None
            assert self.fees is not None
            signed_size = self.filled_size if self.side == "buy" else -self.filled_size
            _check_equal(
                self.after.position,
                self.before.position + signed_size,
                "Filled size differs from position transition",
                scale=max(abs(self.before.position), abs(signed_size)),
            )
            _check_equal(
                self.after.cash,
                self.before.cash - signed_size * self.filled_price - self.fees,
                "Fill and fees differ from cash transition",
                scale=max(abs(self.before.cash), abs(signed_size * self.filled_price)),
            )
        else:
            # execute_order_nb normalizes close-to-zero balances before trying
            # an order, including attempts it later ignores/rejects. Preserve
            # those native values, but permit only this specific dust->0 change.
            # Equity is derived separately above and may change by dust * price.
            for field in ("cash", "position", "debt", "free_cash"):
                before, after = getattr(self.before, field), getattr(self.after, field)
                if before != after and not (
                    after == 0.0
                    and math.isclose(before, 0.0, rel_tol=1e-9, abs_tol=1e-12)
                ):
                    raise ValueError("Unfilled execution changed account balances")
        return self


class AccountState(ExecutionModel):
    """End-of-bar state marked at close, including bars without orders."""

    bar_index: int = Field(ge=0)
    timestamp: str | None
    cash: float
    free_cash: float
    position: float
    asset_value: float
    equity: float
    mark_price: float = Field(gt=0)

    @model_validator(mode="after")
    def coherent_value(self) -> Self:
        _check_equal(
            self.asset_value,
            self.position * self.mark_price,
            "Account asset value differs from position marked at close",
        )
        _check_equal(
            self.equity,
            self.cash + self.asset_value,
            "Account equity differs from cash plus asset value",
        )
        return self


def _check_equal(
    actual: float, expected: float, message: str, *, scale: float = 0.0
) -> None:
    # Native all-in fills clip dust balances to zero; compare at transaction
    # scale so subtraction of large notionals does not reject valid fills.
    if not math.isclose(
        actual, expected, rel_tol=1e-10, abs_tol=max(1e-8, scale * 1e-10)
    ):
        raise ValueError(message)


def _enum_name(enum: Any, value: Any) -> str:
    index = int(value)
    if not 0 <= index < len(enum._fields):
        raise ValueError("Unknown native execution enum")
    return str(enum._fields[index]).lower()


def _snapshot(log: dict[str, Any], prefix: str, price: float) -> AccountSnapshot:
    cash, position = float(log[f"{prefix}cash"]), float(log[f"{prefix}position"])
    return AccountSnapshot(
        cash=cash,
        position=position,
        free_cash=float(log[f"{prefix}free_cash"]),
        debt=float(log[f"{prefix}debt"]),
        equity=cash + position * price,
    )


def _optional_price(bar: pd.Series, name: str) -> float | None:
    value = bar.get(name)
    return None if value is None or pd.isna(value) else float(value)


def _request_size(value: float) -> tuple[float | None, str]:
    if math.isfinite(value):
        return value, "finite"
    if math.isnan(value):
        return None, "not_a_number"
    return None, "positive_infinity" if value > 0 else "negative_infinity"


def extract_execution_records(
    portfolio: vbt.Portfolio,
    market: pd.DataFrame,
    timestamps: pd.DatetimeIndex | None,
) -> tuple[list[ExecutionRecord], list[AccountState]]:
    """Copy only observations produced by this portfolio; never rerun simulation."""
    from vectorbt.portfolio.enums import (
        Direction,
        OrderSide,
        OrderStatus,
        OrderStatusInfo,
        SizeType,
    )

    times: list[str | None] = (
        [timestamp.isoformat() for timestamp in timestamps]
        if timestamps is not None
        else [None] * len(market)
    )
    orders = {
        int(order["id"]): order
        for order in portfolio.orders.records.to_dict(orient="records")
    }
    records: list[ExecutionRecord] = []
    matched: set[int] = set()
    for log in portfolio.logs.records.to_dict(orient="records"):
        bar_index = int(log["idx"])
        bar = market.iloc[bar_index]
        native_order_id = int(log["order_id"])
        filled = int(log["res_status"]) == 0
        order = orders.get(native_order_id) if filled else None
        if filled:
            if order is None or native_order_id in matched:
                raise ValueError("Execution log has no unique native order")
            if int(order["idx"]) != bar_index or int(order["col"]) != int(log["col"]):
                raise ValueError("Execution log and native order locations differ")
            for key in ("size", "price", "fees", "side"):
                _check_equal(
                    float(order[key]),
                    float(log[f"res_{key}"]),
                    "Execution log and native order fill differ",
                )
            matched.add(native_order_id)
        elif native_order_id != -1:
            raise ValueError("Unfilled native log unexpectedly references an order")
        requested_size, size_kind = _request_size(float(log["req_size"]))
        valuation_price = float(log["req_price"])
        records.append(
            ExecutionRecord.model_validate(
                {
                    "execution_id": int(log["id"]),
                    "order_id": native_order_id if filled else None,
                    "bar_index": bar_index,
                    "timestamp": times[bar_index],
                    "status": _enum_name(OrderStatus, log["res_status"]),
                    "status_info": _enum_name(OrderStatusInfo, log["res_status_info"])
                    if int(log["res_status_info"]) != -1
                    else None,
                    "side": _enum_name(OrderSide, order["side"])
                    if order is not None
                    else None,
                    "requested_size": requested_size,
                    "requested_size_kind": size_kind,
                    "requested_size_type": _enum_name(SizeType, log["req_size_type"]),
                    "requested_direction": _enum_name(Direction, log["req_direction"]),
                    "requested_price": valuation_price,
                    "requested_fee_rate": float(log["req_fees"]),
                    "requested_fixed_fees": float(log["req_fixed_fees"]),
                    "requested_slippage": float(log["req_slippage"]),
                    "filled_size": float(order["size"]) if order is not None else None,
                    "filled_price": float(order["price"])
                    if order is not None
                    else None,
                    "fees": float(order["fees"]) if order is not None else None,
                    "market": {
                        "open": _optional_price(bar, "open"),
                        "high": _optional_price(bar, "high"),
                        "low": _optional_price(bar, "low"),
                        "close": float(bar["close"]),
                    },
                    "valuation_price": valuation_price,
                    "before": _snapshot(log, "", valuation_price),
                    "after": _snapshot(log, "new_", valuation_price),
                }
            )
        )
    if matched != orders.keys():
        raise ValueError("Native orders are missing execution logs")

    arrays = {
        "cash": np.asarray(portfolio.cash(), dtype=float),
        "free_cash": np.asarray(portfolio.cash(free=True), dtype=float),
        "position": np.asarray(portfolio.assets(), dtype=float),
        "asset_value": np.asarray(portfolio.asset_value(), dtype=float),
        "equity": np.asarray(portfolio.value(), dtype=float),
        "mark_price": market["close"].to_numpy(dtype=float),
    }
    states = [
        AccountState.model_validate(
            {
                "bar_index": index,
                "timestamp": times[index],
                **{name: float(values[index]) for name, values in arrays.items()},
            }
        )
        for index in range(len(market))
    ]
    return records, states
