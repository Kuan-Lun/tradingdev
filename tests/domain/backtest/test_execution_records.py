"""Order/account observations preserve native fills without changing simulation."""

from __future__ import annotations

import json
import math
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import pytest
import vectorbt as vbt
from pydantic import ValidationError

from tradingdev.adapters.storage.filesystem import WorkspacePaths, sha256_file
from tradingdev.adapters.storage.performance import (
    PerformanceArtifactError,
    PerformanceStore,
)
from tradingdev.adapters.storage.sqlite import SQLiteStore
from tradingdev.domain.backtest.execution_records import (
    ExecutionRecord,
    extract_execution_records,
)
from tradingdev.domain.backtest.metrics import calculate_metrics, normalized_trades
from tradingdev.domain.backtest.signal_engine import SignalBacktestEngine
from tradingdev.domain.backtest.volume_engine import VolumeBacktestEngine
from tradingdev.domain.performance.artifacts import (
    ScopeObservations,
    build_artifacts,
    scope_from_backtest,
)

if TYPE_CHECKING:
    from pathlib import Path


def _market(signals: list[int] | None = None) -> pd.DataFrame:
    signals = [1, -1, 0, 1, 1] if signals is None else signals
    prices = np.array([100.0, 110.0, 90.0, 95.0, 105.0])[: len(signals)]
    return pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2024-01-01", periods=len(signals), tz="Asia/Taipei"
            ),
            "open": prices,
            "high": prices + 10,
            "low": prices - 10,
            "close": prices + 5,
            "signal": signals,
        }
    )


def _engine(**kwargs: Any) -> SignalBacktestEngine:
    return SignalBacktestEngine(init_cash=1000.0, fees=0.001, slippage=0.01, **kwargs)


def test_buy_reverse_close_and_open_final_have_native_fill_accounting() -> None:
    result = _engine().run(_market())
    records, accounts = result.execution_records, result.account_history
    assert records is not None and accounts is not None
    assert [record.side for record in records] == ["buy", "sell", "buy", "buy"]
    assert [record.bar_index for record in records] == [1, 2, 3, 4]
    assert [record.order_id for record in records] == [0, 1, 2, 3]
    assert all(record.valuation_price == record.requested_price for record in records)
    first, reverse, close, last = records
    assert first.timestamp == "2024-01-01T16:00:00+00:00"
    assert first.market.open == first.requested_price == 110.0
    assert first.market.high == 120.0
    assert first.market.low == 100.0
    assert first.market.close == 115.0
    assert first.filled_price == pytest.approx(111.1)
    assert first.filled_size is not None and first.fees is not None
    assert first.fees == pytest.approx(first.filled_size * 111.1 * 0.001)
    assert first.before.cash == first.before.equity == 1000.0
    assert first.before.position == 0.0
    assert first.after.position == first.filled_size
    assert first.after.cash == pytest.approx(0.0)
    # Marking both sides at the request price makes slippage and fees visible.
    assert first.after.equity == pytest.approx(first.filled_size * 110.0)
    assert first.after.equity < first.before.equity - first.fees
    assert reverse.before.position > 0 > reverse.after.position
    assert reverse.filled_size == pytest.approx(
        reverse.before.position + abs(reverse.after.position)
    )
    assert reverse.after.cash > reverse.after.equity
    assert reverse.after.debt > 0
    assert close.after.position == 0.0
    assert last.after.position > 0
    assert result.trades[-1]["status"] == "open"
    assert len(records) == 4  # final mark is not an invented exit order
    assert len(accounts) == len(result.equity_curve) == 5
    assert accounts[1].position == first.after.position
    assert accounts[1].mark_price == 115.0
    assert accounts[1].equity != first.after.equity
    np.testing.assert_array_equal(
        [state.equity for state in accounts], result.equity_curve
    )
    json.dumps([record.model_dump(mode="json") for record in records], allow_nan=False)


@pytest.mark.parametrize(
    "settings",
    [{}, {"position_size": 250.0}, {"stop_loss": 0.05}, {"take_profit": 0.03}],
)
def test_logging_does_not_change_orders_values_trades_or_metrics(
    monkeypatch: pytest.MonkeyPatch, settings: dict[str, float]
) -> None:
    original = vbt.Portfolio.from_signals
    native: list[Any] = []

    def compare(**kwargs: Any) -> Any:
        assert kwargs["log"] is True
        assert "update_value" not in kwargs
        logged = original(**kwargs)
        baseline = original(**{**kwargs, "log": False})
        np.testing.assert_array_equal(
            logged.orders.records_arr, baseline.orders.records_arr
        )
        np.testing.assert_array_equal(logged.value(), baseline.value())
        assert normalized_trades(logged.trades) == normalized_trades(baseline.trades)
        native.extend([logged, baseline])
        return logged

    monkeypatch.setattr(vbt.Portfolio, "from_signals", compare)
    result = _engine(**settings).run(_market())
    assert (
        result.metrics
        == calculate_metrics(
            native[1], timestamps=native[1].wrapper.index, frequency="1h"
        ).metrics
    )
    assert result.execution_records
    # VectorBT's default new_value is stale after fills; our equity is not.
    assert (
        native[0].logs.records.iloc[0]["new_value"]
        != result.execution_records[0].after.equity
    )


@pytest.mark.parametrize("empty", [False, True])
def test_available_no_orders_is_empty_including_empty_market(empty: bool) -> None:
    result = _engine().run(_market([] if empty else [0, 0, 0]))
    scope, observations = scope_from_backtest(result)
    assert observations.execution_records == []
    assert observations.account_history is not None
    assert len(observations.account_history) == (0 if empty else 3)
    metadata = scope.metadata["observations"]
    assert isinstance(metadata, dict)
    assert metadata["execution_records_availability"] == "available"


def test_missing_timestamps_or_ohlc_are_not_fabricated() -> None:
    result = _engine().run(_market().drop(columns=["timestamp", "open", "high", "low"]))
    _, observations = scope_from_backtest(result)
    assert observations.timestamps is None
    assert observations.execution_records
    first = observations.execution_records[0]
    assert first.timestamp is None
    assert first.market.open is first.market.high is first.market.low is None
    assert first.requested_price == first.market.close
    assert observations.account_history
    assert all(state.timestamp is None for state in observations.account_history)


def test_large_all_in_account_tolerates_native_rounding_of_dust_cash() -> None:
    result = SignalBacktestEngine(
        init_cash=10_000_000_000.0, fees=0.001, slippage=0.01
    ).run(_market())
    _, observations = scope_from_backtest(result)
    assert observations.execution_records
    assert observations.execution_records[-1].after.cash == pytest.approx(0.0)


def test_rejected_native_attempt_has_no_fill_and_preserves_balances() -> None:
    market = _market().set_index("timestamp")
    portfolio = vbt.Portfolio.from_signals(
        market["close"],
        entries=True,
        init_cash=1000.0,
        price=market["open"],
        reject_prob=1.0,
        log=True,
    )
    records, accounts = extract_execution_records(
        portfolio, market, pd.DatetimeIndex(market.index)
    )
    assert len(records) == len(market)
    for record in records:
        assert record.status == "rejected"
        assert record.status_info == "randomevent"
        assert (
            record.order_id
            is record.filled_size
            is record.filled_price
            is record.fees
            is None
        )
        assert record.before == record.after
    assert all(state.equity == 1000.0 and state.position == 0.0 for state in accounts)


def test_unfilled_native_attempt_preserves_dust_normalization_without_failing() -> None:
    result = SignalBacktestEngine(init_cash=1e-12, fees=0.0, slippage=0.0).run(
        pd.DataFrame({"close": [100.0] * 3, "signal": [1, 0, 0]})
    )
    _, observations = scope_from_backtest(result)
    assert observations.execution_records
    record = observations.execution_records[0]
    assert record.status == "rejected" and record.filled_size is None
    assert (
        record.before.cash == record.before.free_cash == record.before.equity == 1e-12
    )
    assert record.after.cash == record.after.free_cash == record.after.equity == 0.0
    assert record.before.position == record.after.position == 0.0
    # Native per-bar cash reconstruction also normalizes dust independently;
    # keep the original pre-attempt amount in the log instead of replacing it.
    assert observations.account_history
    assert observations.account_history[1].cash == 0.0
    assert ExecutionRecord.model_validate_json(record.model_dump_json()) == record


@pytest.mark.parametrize(
    ("field", "before", "after"),
    [
        ("cash", 1000.0, 999.0),
        ("cash", 1000.0, 1000.0 + 1e-7),
        ("cash", 2e-12, 0.0),
        ("cash", 0.0, 1e-12),
        ("position", 2e-12, 0.0),
        ("debt", 0.0, 1.0),
        ("free_cash", 1000.0, 999.0),
    ],
)
def test_unfilled_validation_rejects_changes_other_than_native_dust_to_zero(
    field: str, before: float, after: float
) -> None:
    market = _market().set_index("timestamp")
    portfolio = vbt.Portfolio.from_signals(
        market["close"],
        entries=True,
        init_cash=1000.0,
        price=market["open"],
        reject_prob=1.0,
        log=True,
    )
    records, _ = extract_execution_records(
        portfolio, market, pd.DatetimeIndex(market.index)
    )
    payload = records[0].model_dump(mode="json")
    payload["before"][field] = before
    payload["after"][field] = after
    for side in ("before", "after"):
        state = payload[side]
        state["equity"] = state["cash"] + state["position"] * payload["valuation_price"]
    with pytest.raises(ValidationError, match="Unfilled execution changed"):
        ExecutionRecord.model_validate(payload)


def test_orders_without_logs_are_rejected_instead_of_silently_omitted() -> None:
    market = _market().set_index("timestamp")
    portfolio = vbt.Portfolio.from_signals(market["close"], entries=True, log=False)
    with pytest.raises(ValueError, match="missing execution logs"):
        extract_execution_records(portfolio, market, pd.DatetimeIndex(market.index))


def test_legacy_fields_and_volume_accounting_stay_unavailable() -> None:
    result = _engine().run(_market())
    _, observations = scope_from_backtest(result)
    legacy = observations.model_dump(mode="json")
    legacy.pop("execution_records")
    legacy.pop("account_history")
    restored = ScopeObservations.model_validate_json(json.dumps(legacy))
    assert restored.execution_records is None
    assert restored.account_history is None
    result.execution_records = result.account_history = None
    scope, _ = scope_from_backtest(result)
    metadata = scope.metadata["observations"]
    assert isinstance(metadata, dict)
    assert metadata["account_history_availability"] == "not_recorded"
    volume = VolumeBacktestEngine(position_size=200.0).run(_market())
    scope, observations = scope_from_backtest(volume)
    assert observations.execution_records is None
    assert observations.account_history is None
    metadata = scope.metadata["observations"]
    assert isinstance(metadata, dict)
    assert metadata["account_history_availability"] == "unsupported_volume_accounting"


@pytest.mark.parametrize(
    "corruption",
    [
        "negative_id",
        "boolean_id",
        "nan",
        "timestamp",
        "bar_index",
        "duplicate_id",
        "order_id",
        "account_length",
        "account_bar",
        "account_timestamp",
        "account_equity",
        "fill_position",
        "fill_cash",
        "unfilled_order",
    ],
)
def test_observation_validation_rejects_untrustworthy_execution_data(
    corruption: str,
) -> None:
    _, observations = scope_from_backtest(_engine().run(_market()))
    payload = observations.model_dump(mode="json")
    records, states = payload["execution_records"], payload["account_history"]
    first = records[0]
    if corruption == "negative_id":
        first["execution_id"] = -1
    elif corruption == "boolean_id":
        first["order_id"] = True
    elif corruption == "nan":
        first["fees"] = float("nan")
    elif corruption == "timestamp":
        first["timestamp"] = "2024-01-01T00:00:00+00:00"
    elif corruption == "bar_index":
        first["bar_index"] = 999
    elif corruption == "duplicate_id":
        records[1]["execution_id"] = first["execution_id"]
    elif corruption == "order_id":
        records[1]["order_id"] = first["order_id"]
    elif corruption == "account_length":
        states.pop()
    elif corruption == "account_bar":
        states[0]["bar_index"] = 1
    elif corruption == "account_timestamp":
        states[0]["timestamp"] = None
    elif corruption == "account_equity":
        payload["equity_curve"][0] += 1.0
    elif corruption == "fill_position":
        first["filled_size"] += 1.0
    elif corruption == "fill_cash":
        first["fees"] += 1.0
    else:
        first["status"] = "rejected"
    with pytest.raises(ValidationError):
        ScopeObservations.model_validate(payload)


def _change_valuation_price(record: dict[str, Any], price: float) -> None:
    record["valuation_price"] = price
    for side in ("before", "after"):
        state = record[side]
        state["equity"] = state["cash"] + state["position"] * price


@pytest.mark.parametrize("nearby", [False, True], ids=["different_mark", "one_ulp"])
def test_execution_valuation_must_equal_request_even_with_consistent_equity(
    nearby: bool,
) -> None:
    _, observations = scope_from_backtest(_engine().run(_market()))
    payload = observations.model_dump(mode="json")
    record = payload["execution_records"][0]
    price = record["requested_price"]
    _change_valuation_price(
        record, math.nextafter(price, math.inf) if nearby else price * 2.0
    )
    # Both equity snapshots remain arithmetically consistent with the wrong mark.
    message = "Execution valuation price differs from requested price"
    with pytest.raises(ValidationError, match=message):
        ExecutionRecord.model_validate(record)
    with pytest.raises(ValidationError, match=message):
        ScopeObservations.model_validate(payload)


def test_saved_observations_reject_different_execution_mark_with_valid_sha(
    tmp_path: Path,
) -> None:
    result = _engine().run(_market())
    scope, observations = scope_from_backtest(result)
    workspace = WorkspacePaths(tmp_path)
    store = SQLiteStore(workspace)
    store.create_run(
        run_id="ledger",
        job_id="ledger",
        strategy_id="fixture",
        artifact_dir=workspace.runs / "ledger",
        metrics=result.metrics,
    )
    details = PerformanceStore(workspace, store)
    details.publish(
        build_artifacts("ledger", None, "full", {"full": scope}, {"full": observations})
    )
    assert details.load_observations("ledger").scopes["full"] == observations
    path = workspace.runs / "ledger" / "observations.json"
    payload = json.loads(path.read_bytes())
    record = payload["scopes"]["full"]["execution_records"][0]
    _change_valuation_price(record, record["requested_price"] * 2.0)
    path.write_text(json.dumps(payload), encoding="utf-8")
    # A matching registry digest must not substitute for semantic validation.
    with store.connect() as connection:
        connection.execute(
            "UPDATE artifacts SET sha256 = ? WHERE artifact_id = ?",
            (sha256_file(path), "ledger:observations_json"),
        )
    with pytest.raises(
        PerformanceArtifactError,
        match="Execution valuation price differs from requested price",
    ) as error:
        details.load_observations("ledger")
    assert error.value.code == "performance_artifact_invalid"


def test_json_publication_and_reload_preserve_execution_records_without_pickle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = _engine().run(_market())
    scope, observations = scope_from_backtest(result)
    artifacts = build_artifacts(
        "ledger", None, "full", {"full": scope}, {"full": observations}
    )
    workspace = WorkspacePaths(tmp_path)
    store = SQLiteStore(workspace)
    store.create_run(
        run_id="ledger",
        job_id="ledger",
        strategy_id="fixture",
        artifact_dir=workspace.runs / "ledger",
        metrics=result.metrics,
    )
    details = PerformanceStore(workspace, store)
    details.publish(artifacts)
    path = workspace.runs / "ledger" / "observations.json"
    before = path.read_bytes()
    monkeypatch.setattr("pickle.load", lambda *_: pytest.fail("Must read JSON"))
    loaded = details.load_observations("ledger").scopes["full"]
    assert loaded == observations
    assert loaded.execution_records == result.execution_records
    assert loaded.account_history == result.account_history
    assert loaded.execution_records
    snapshot: Any = loaded.execution_records[0].before
    with pytest.raises(ValidationError, match="frozen"):
        snapshot.cash = 1.0
    assert path.read_bytes() == before
