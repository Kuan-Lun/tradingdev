"""Execution/account queries read recorded JSON without reconstructing history."""

from __future__ import annotations

import hashlib
import json
import sys
from typing import TYPE_CHECKING, Any

import pytest

from tradingdev.adapters.storage.execution_manifests import ExecutionManifestStore
from tradingdev.adapters.storage.filesystem import WorkspacePaths, sha256_file
from tradingdev.adapters.storage.performance import PerformanceStore
from tradingdev.adapters.storage.sqlite import SQLiteStore
from tradingdev.app.contracts.history import (
    HistoryQueryError,
    RunAccountHistoryResponse,
    RunExecutionsResponse,
)
from tradingdev.app.trade_history_service import TradeHistoryService
from tradingdev.domain.backtest.execution_records import AccountState, ExecutionRecord
from tradingdev.domain.execution import ExecutionManifest, OptimizationSpec
from tradingdev.domain.performance.artifacts import (
    FoldStats,
    PerformanceScope,
    ScopeObservations,
    build_artifacts,
)
from tradingdev.domain.strategies.execution import StrategyExecution

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def storage(tmp_path: Path) -> tuple[WorkspacePaths, SQLiteStore, TradeHistoryService]:
    workspace = WorkspacePaths(tmp_path / "workspace")
    store = SQLiteStore(workspace)
    return workspace, store, TradeHistoryService(workspace=workspace, store=store)


def _observations(*, timestamps: bool = True, empty: bool = False) -> ScopeObservations:
    times = [
        "2024-01-01T08:00:00.000000000+08:00",
        "2024-01-02T08:00:00.000000000+08:00",
        "2024-01-03T07:59:59.999999999+08:00",
        "2024-01-03T08:00:00.000000000+08:00",
    ]
    records: list[ExecutionRecord] = []
    states = []
    cash, position, free_cash, debt = 1000.0, 0.0, 1000.0, 0.0
    # A buy, a reversal, two unsuccessful attempts, and a closing buy.
    orders = [(0, "buy", 2.0), (1, "sell", 3.0), (3, "buy", 1.0)]
    for index in range(4):
        price = 100.0 + index * 10.0
        before = {
            "cash": cash,
            "position": position,
            "free_cash": free_cash,
            "debt": debt,
            "equity": cash + position * price,
        }
        attempts = ["rejected", "ignored"] if index == 2 else ["filled"]
        for status in attempts:
            order = next((row for row in orders if row[0] == index), None)
            side, size, order_id = None, None, None
            if order is not None:
                _, side, size = order
                order_id = orders.index(order)
                signed_size = size if side == "buy" else -size
                cash -= signed_size * price + 1.0
                position += signed_size
                debt = 220.0 if position < 0 else 0.0
                free_cash = cash - debt
            after = {
                "cash": cash,
                "position": position,
                "free_cash": free_cash,
                "debt": debt,
                "equity": cash + position * price,
            }
            records.append(
                ExecutionRecord.model_validate(
                    {
                        "execution_id": len(records) + 4,
                        "order_id": order_id,
                        "bar_index": index,
                        "timestamp": times[index] if timestamps else None,
                        "status": status,
                        "status_info": None if order is not None else "sizezero",
                        "side": side,
                        "requested_size": size,
                        "requested_size_kind": "finite"
                        if size is not None
                        else "positive_infinity",
                        "requested_size_type": "amount",
                        "requested_direction": "both",
                        "requested_price": price,
                        "requested_fee_rate": 0.0,
                        "requested_fixed_fees": 1.0,
                        "requested_slippage": 0.0,
                        "filled_size": size,
                        "filled_price": price if order is not None else None,
                        "fees": 1.0 if order is not None else None,
                        "market": {
                            "open": price,
                            "high": price + 10,
                            "low": price - 10,
                            "close": price + 5,
                        },
                        "valuation_price": price,
                        "before": before,
                        "after": after,
                    }
                )
            )
        states.append(
            AccountState(
                bar_index=index,
                timestamp=times[index] if timestamps else None,
                cash=cash,
                free_cash=free_cash,
                position=position,
                asset_value=position * (price + 5),
                equity=cash + position * (price + 5),
                mark_price=price + 5,
            )
        )
    return ScopeObservations(
        init_cash=1000.0,
        equity_curve=[] if empty else [state.equity for state in states],
        returns=[] if empty else [None] * 4,
        timestamps=([] if empty else times) if timestamps else None,
        trades=[],
        execution_records=[] if empty else records,
        account_history=[] if empty else states,
    )


def _publish(
    storage: tuple[WorkspacePaths, SQLiteStore, TradeHistoryService],
    *,
    availability: str = "available",
    timestamps: bool = True,
    empty: bool = False,
    legacy: bool = False,
    summary: bool = False,
) -> None:
    workspace, store, _ = storage
    volume = availability == "unsupported_volume_accounting"
    observations = _observations(timestamps=timestamps, empty=empty)
    if availability != "available":
        observations = observations.model_copy(
            update={"execution_records": None, "account_history": None}
        )
    if volume:
        observations = observations.model_copy(
            update={
                "init_cash": None,
                "returns": None,
                "equity_curve": [
                    value - 1000.0
                    for value in observations.equity_curve
                    if value is not None
                ],
            }
        )
    manifest = ExecutionManifest.create(
        kind="optimization",
        config={
            "strategy": {"id": "macd"},
            "backtest": {
                "symbol": "BTC/USDT",
                "timeframe": "1d",
                "start_date": "2024-01-01",
                "end_date": "2025-01-03",
                "mode": "volume" if volume else "signal",
                "init_cash": None if volume else 1000.0,
            },
        },
        strategy_execution=StrategyExecution(
            kind="generated", constructor_kwargs={"fast": 12, "slow": 26}
        ),
        optimization=OptimizationSpec(
            param_ranges={"fast": [10, 12]},
            optimization_metric="total_pnl",
            direction="maximize",
            train_start="2024-01-01",
            train_end="2024-01-03",
            test_start="2025-01-01",
            test_end="2025-01-03",
        ),
    )
    manifest_hash = None if legacy else manifest.manifest_hash
    store.create_run(
        run_id="run",
        job_id="job",
        strategy_id="macd",
        artifact_dir=workspace.runs / "run",
        manifest_hash=manifest_hash,
        metrics={"total_pnl": -3.0},
    )
    if not legacy:
        path = ExecutionManifestStore(workspace).publish("run", manifest)
        store.create_artifact(
            artifact_id="run:execution_manifest",
            run_id="run",
            artifact_type="execution_manifest",
            path=path,
            sha256=sha256_file(path),
        )
    scope = PerformanceScope(
        mode="volume" if volume else "signal",
        values={"total_pnl": -3.0},
        parameters={"fast": 10},
        metadata={},
    )
    scopes = {"trial/0/train": scope}
    default = "trial/0/train"
    if summary:
        scopes["test_summary"] = PerformanceScope(
            kind="fold_summary",
            mode="signal",
            values={
                "total_pnl": FoldStats(
                    mean=-3.0, std=0.0, min=-3.0, max=-3.0, valid_count=1
                )
            },
            metadata={},
        )
        default = "test_summary"
    PerformanceStore(workspace, store).publish(
        build_artifacts(
            "run", manifest_hash, default, scopes, {"trial/0/train": observations}
        )
    )


def test_execution_filters_page_original_ids_and_keep_full_accounting(
    storage: tuple[WorkspacePaths, SQLiteStore, TradeHistoryService],
) -> None:
    _publish(storage)
    service = storage[2]
    first = RunExecutionsResponse.model_validate(
        service.get_run_executions("run", limit=1)
    )
    assert (first.total, first.matched, first.next_offset) == (5, 5, 1)
    assert first.availability == "available" and first.accounting == "vectorbt_generic"
    assert first.init_cash == 1000.0
    assert first.parameters == {"fast": 10, "slow": 26}
    assert first.parameter_provenance == "execution_manifest_and_trial"
    assert first.parameters_complete
    record = first.records[0]
    assert (record.execution_id, record.order_id, record.bar_index) == (4, 0, 0)
    assert record.timestamp == "2024-01-01T00:00:00+00:00"
    assert record.before.cash == 1000.0 and record.after.cash == 799.0
    assert record.before.position == 0.0 and record.after.position == 2.0
    assert record.market.close == 105.0 and record.filled_price == 100.0
    assert record.fees == 1.0
    reversal = RunExecutionsResponse.model_validate(
        service.get_run_executions("run", side="sell", status="filled")
    )
    assert reversal.matched == 1 and reversal.total == 5
    assert reversal.records[0].execution_id == 5
    assert reversal.records[0].before.position > 0 > reversal.records[0].after.position
    assert reversal.records[0].filled_size == 3.0
    assert service.get_run_executions("run", offset=5)["records"] == []


@pytest.mark.parametrize("status,identity", [("rejected", 6), ("ignored", 7)])
def test_unfilled_attempts_retain_request_but_never_invent_a_fill(
    storage: tuple[WorkspacePaths, SQLiteStore, TradeHistoryService],
    status: str,
    identity: int,
) -> None:
    _publish(storage)
    service = storage[2]
    page = RunExecutionsResponse.model_validate(
        service.get_run_executions("run", status=status)
    )
    assert page.matched == 1
    record = page.records[0]
    assert record.execution_id == identity
    assert record.requested_size is None
    assert record.requested_size_kind == "positive_infinity"
    assert record.order_id is None and record.side is None
    assert record.filled_price is None and record.fees is None
    assert record.before == record.after
    assert service.get_run_executions("run", status=status, side="buy")["matched"] == 0


def test_dates_are_utc_inclusive_and_close_states_include_unfilled_bars(
    storage: tuple[WorkspacePaths, SQLiteStore, TradeHistoryService],
) -> None:
    _publish(storage)
    service = storage[2]
    date: dict[str, Any] = {"start": "2024-01-02", "end": "2024-01-02"}
    attempts = RunExecutionsResponse.model_validate(
        service.get_run_executions("run", **date)
    )
    assert [record.execution_id for record in attempts.records] == [5, 6, 7]
    assert attempts.records[-1].timestamp == "2024-01-02T23:59:59.999999999+00:00"
    assert (
        service.get_run_executions("run", end="2024-01-02T08:00:00+08:00")["matched"]
        == 2
    )
    page = RunAccountHistoryResponse.model_validate(
        service.get_run_account_history("run", limit=1, offset=1, **date)
    )
    assert (page.total, page.matched, page.next_offset) == (4, 2, None)
    assert page.states[0].bar_index == 2
    assert page.states[0].timestamp == "2024-01-02T23:59:59.999999999+00:00"
    assert page.valuation_basis == "bar_close"
    assert page.timestamp_semantics == "bar_timestamp_end_of_bar_state"
    assert page.states[0].equity == 1003.0
    assert page.states[0].position == -1.0
    assert page.states[0].asset_value == -125.0
    assert attempts.records[-1].after.equity == 1008.0
    assert service.get_run_account_history("run", offset=4)["states"] == []


@pytest.mark.parametrize(
    "availability", ["not_recorded", "unsupported_volume_accounting"]
)
@pytest.mark.parametrize("legacy", [False, True])
def test_absent_history_remains_explicit_even_with_dates_and_no_timestamps(
    storage: tuple[WorkspacePaths, SQLiteStore, TradeHistoryService],
    availability: str,
    legacy: bool,
) -> None:
    _publish(storage, availability=availability, timestamps=False, legacy=legacy)
    executions = RunExecutionsResponse.model_validate(
        storage[2].get_run_executions("run", start="2024-01-01")
    )
    accounts = RunAccountHistoryResponse.model_validate(
        storage[2].get_run_account_history("run", end="2024-01-02")
    )
    for page in (executions, accounts):
        assert page.availability == availability
        assert page.accounting == (
            None
            if availability == "unsupported_volume_accounting"
            else "vectorbt_generic"
        )
        assert page.total == page.matched == 0 and page.next_offset is None
        assert page.parameters_complete is not legacy
    assert executions.records == [] and accounts.states == []


def test_new_recorded_empty_history_is_available(
    storage: tuple[WorkspacePaths, SQLiteStore, TradeHistoryService],
) -> None:
    _publish(storage, empty=True)
    executions = RunExecutionsResponse.model_validate(
        storage[2].get_run_executions("run")
    )
    accounts = RunAccountHistoryResponse.model_validate(
        storage[2].get_run_account_history("run")
    )
    assert executions.availability == accounts.availability == "available"
    assert executions.records == [] and accounts.states == []


def test_old_json_without_optional_ledger_fields_is_not_reconstructed(
    storage: tuple[WorkspacePaths, SQLiteStore, TradeHistoryService],
) -> None:
    _publish(storage, availability="not_recorded")
    workspace, store, service = storage
    path = workspace.runs / "run" / "observations.json"
    payload = json.loads(path.read_text())
    del payload["scopes"]["trial/0/train"]["execution_records"]
    del payload["scopes"]["trial/0/train"]["account_history"]
    path.write_text(json.dumps(payload))
    with store.connect() as connection:
        connection.execute(
            "update artifacts set sha256 = ? where artifact_id = ?",
            (sha256_file(path), "run:observations_json"),
        )
    executions = RunExecutionsResponse.model_validate(service.get_run_executions("run"))
    accounts = RunAccountHistoryResponse.model_validate(
        service.get_run_account_history("run")
    )
    assert executions.availability == accounts.availability == "not_recorded"
    assert executions.records == [] and accounts.states == []


@pytest.mark.parametrize("method", ["get_run_executions", "get_run_account_history"])
def test_scope_and_timestamp_failures_are_explicit(
    storage: tuple[WorkspacePaths, SQLiteStore, TradeHistoryService], method: str
) -> None:
    _publish(storage, timestamps=False, summary=True)
    query = getattr(storage[2], method)
    assert query("absent")["code"] == "run_not_found"
    for scope, code in [
        (None, "scope_has_no_observations"),
        ("unknown", "unknown_history_scope"),
    ]:
        error = HistoryQueryError.model_validate(query("run", scope=scope))
        assert error.code == code and error.available_scopes == ["trial/0/train"]
    assert query("run", scope="trial/0/train")["success"]
    assert (
        query("run", scope="trial/0/train", start="2024-01-01")["code"]
        == "timestamps_unavailable"
    )


@pytest.mark.parametrize("method", ["get_run_executions", "get_run_account_history"])
@pytest.mark.parametrize(
    "arguments",
    [
        {"offset": -1},
        {"offset": True},
        {"limit": 0},
        {"limit": 501},
        {"limit": 1.5},
        {"start": "invalid"},
        {"end": "2024-99-99"},
        {"start": "2024-01-03", "end": "2024-01-02"},
    ],
)
def test_invalid_queries_fail_before_reading_a_run(
    storage: tuple[WorkspacePaths, SQLiteStore, TradeHistoryService],
    method: str,
    arguments: dict[str, Any],
) -> None:
    error = HistoryQueryError.model_validate(
        getattr(storage[2], method)("absent", **arguments)
    )
    assert error.code == "invalid_history_query"


@pytest.mark.parametrize("arguments", [{"status": "open"}, {"side": "long"}])
def test_execution_enums_are_validated_by_the_application(
    storage: tuple[WorkspacePaths, SQLiteStore, TradeHistoryService],
    arguments: dict[str, Any],
) -> None:
    assert (
        storage[2].get_run_executions("absent", **arguments)["code"]
        == "invalid_history_query"
    )


@pytest.mark.parametrize("damage", ["hash", "manifest", "alignment"])
def test_both_queries_enforce_saved_artifact_integrity(
    storage: tuple[WorkspacePaths, SQLiteStore, TradeHistoryService], damage: str
) -> None:
    _publish(storage)
    workspace, store, service = storage
    path = (
        workspace.runs
        / "run"
        / ("manifest.json" if damage == "manifest" else "observations.json")
    )
    payload = json.loads(path.read_text())
    expected = "performance_artifact_invalid"
    if damage == "manifest":
        payload["strategy_execution"]["constructor_kwargs"]["fast"] = 99
        expected = "execution_manifest_invalid"
    else:
        payload["scopes"]["trial/0/train"]["execution_records"][0]["bar_index"] = 2
    path.write_text(json.dumps(payload))
    if damage == "alignment":
        # With matching file digest, semantic validation must still reject shifted data.
        with store.connect() as connection:
            connection.execute(
                "update artifacts set sha256 = ? where artifact_id = ?",
                (
                    hashlib.sha256(path.read_bytes()).hexdigest(),
                    "run:observations_json",
                ),
            )
    assert service.get_run_executions("run")["code"] == expected
    assert service.get_run_account_history("run")["code"] == expected


def test_queries_are_read_only_detached_and_never_replay(
    storage: tuple[WorkspacePaths, SQLiteStore, TradeHistoryService],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _publish(storage)
    workspace, _, service = storage
    before = {
        path: path.read_bytes() for path in (workspace.runs / "run").glob("*.json")
    }

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError(
            "Historical queries must not unpickle or execute a strategy"
        )

    monkeypatch.setattr("pickle.load", forbidden)
    for module in (
        "tradingdev.domain.strategies.loader",
        "tradingdev.domain.backtest.engines",
        "tradingdev.app.backtest_service",
    ):
        monkeypatch.setitem(sys.modules, module, None)
    service.get_run_executions("run")["records"][0]["before"]["cash"] = 0.0
    service.get_run_account_history("run")["states"][0]["position"] = 100.0
    assert service.get_run_executions("run")["records"][0]["before"]["cash"] == 1000.0
    assert service.get_run_account_history("run")["states"][0]["position"] == 2.0
    assert before == {path: path.read_bytes() for path in before}
