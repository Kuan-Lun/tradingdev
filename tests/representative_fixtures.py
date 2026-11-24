"""Deterministic saved sources with complete manifest/scope identity."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Any

from tests.strategy_fixtures import GENERATED_PROGRAM
from tradingdev.adapters.storage.execution_manifests import ExecutionManifestStore
from tradingdev.adapters.storage.performance import PerformanceStore
from tradingdev.domain.execution import ExecutionManifest, OptimizationSpec
from tradingdev.domain.performance.artifacts import (
    FoldStats,
    PerformanceScope,
    ScopeObservations,
    build_artifacts,
)
from tradingdev.domain.research_scope import optimization_backtest
from tradingdev.domain.strategies.execution import StrategyExecution

if TYPE_CHECKING:
    from tradingdev.adapters.storage.filesystem import WorkspacePaths
    from tradingdev.adapters.storage.sqlite import SQLiteStore

SOURCE = b"# historical source; never imported\n"
SOURCE_HASH = hashlib.sha256(SOURCE).hexdigest()


def publish_representative_run(
    workspace: WorkspacePaths,
    store: SQLiteStore,
    run_id: str,
    *,
    window: int = 10,
    revision: str | None = "revision-1",
    strategy_id: str = "strategy-S",
    value: float | None = 1.0,
    bundled: bool = False,
    optimization: bool = False,
    walk_forward: bool = False,
    fees: float = 0.001,
    timeframe: str = "1d",
    start_date: str = "2024-01-01",
    end_date: str = "2024-03-01",
    source: bytes = SOURCE,
) -> ExecutionManifest:
    source_hash = hashlib.sha256(source).hexdigest()
    config: dict[str, Any] = {
        "strategy": {
            "id": strategy_id,
            "revision_id": None if bundled else revision,
            "source_hash": source_hash,
            "parameters": {"window": window},
        },
        "backtest": {
            "symbol": "BTC/USDT",
            "timeframe": timeframe,
            "start_date": start_date,
            "end_date": end_date,
            "init_cash": 1000.0,
            "fees": fees,
            "slippage": 0.0,
            "periods_per_year": 365,
        },
    }
    if walk_forward:
        config["validation"] = {
            "mode": "explicit",
            "train_start": "2024-01-01",
            "train_end": "2024-01-31",
            "test_start": "2024-02-01",
            "test_end": "2024-03-01",
        }
    manifest = ExecutionManifest.create(
        kind="optimization"
        if optimization
        else "walk_forward"
        if walk_forward
        else "backtest",
        config=config,
        strategy_execution=StrategyExecution(
            kind="bundled" if bundled else "generated",
            program=None
            if bundled
            else GENERATED_PROGRAM.model_copy(update={"source_hash": source_hash}),
            constructor_kwargs={"config": {"window": window, "nested": {"default": 7}}}
            if bundled
            else {"window": window, "nested": {"default": 7}},
        ),
        optimization=OptimizationSpec.model_validate(
            {
                "param_ranges": {"window": [10, 20]},
                "optimization_metric": "total_pnl",
                "train_start": "2024-01-01",
                "train_end": "2024-01-31",
                "test_start": "2024-02-01",
                "test_end": "2024-03-01",
            }
        )
        if optimization
        else None,
    )
    scopes = {}
    observations = {}
    names: tuple[str, ...] = (
        ("trial/0/train", "trial/1/train", "test") if optimization else ("full",)
    )
    if walk_forward:
        names = ("fold/0/train", "fold/0/test")
    for index, name in enumerate(names):
        split = "full" if not optimization else "test" if name == "test" else "train"
        if walk_forward:
            split = name.rsplit("/", 1)[-1]
        context = (
            optimization_backtest(
                manifest, "test" if name == "test" else "train"
            ).model_dump(mode="json")
            if optimization
            else manifest.config_copy()["backtest"]
        )
        start = "2024-02-01" if split == "test" else start_date
        parameters = (
            {"window": 10 if index == 0 else 20}
            if optimization
            else {"window": window, "nested": {"default": 7}}
        )
        scopes[name] = PerformanceScope.model_validate(
            {
                "mode": "signal",
                "split": split,
                "trial_index": index if optimization and split == "train" else None,
                "fold_index": 0 if walk_forward else None,
                "parameters": parameters,
                "values": {"sharpe_ratio": value, "total_pnl": float(index + 1)},
                "metadata": {
                    "execution_context": {
                        **context,
                        "random_seed": manifest.config.get("random_seed"),
                    },
                    "unavailable": {"sharpe_ratio": "zero_denominator"}
                    if value is None
                    else {},
                },
            }
        )
        observations[name] = ScopeObservations(
            init_cash=1000.0,
            equity_curve=[1000.0, 1001.0],
            returns=[0.0, 0.001],
            timestamps=[f"{start}T00:00:00Z", f"{start}T12:00:00Z"],
            trades=[],
        )
    if walk_forward:
        scopes["test_summary"] = PerformanceScope(
            kind="fold_summary",
            mode="signal",
            split="test",
            parameters={"window": window, "nested": {"default": 7}},
            values={
                "total_pnl": FoldStats(
                    mean=2.0, std=0.0, min=2.0, max=2.0, valid_count=1
                )
            },
            metadata={"n_folds": 1},
        )
    ExecutionManifestStore(workspace).publish(run_id, manifest)
    path = workspace.runs / run_id / "strategy.py"
    path.write_bytes(source)
    store.create_run(
        run_id=run_id,
        job_id=f"job-{run_id}",
        strategy_id=strategy_id,
        revision_id=None if bundled else revision,
        source_hash=source_hash,
        artifact_dir=path.parent,
        metrics={"sharpe_ratio": value},
        manifest_hash=manifest.manifest_hash,
    )
    store.create_artifact(
        artifact_id=f"{run_id}:strategy_source",
        run_id=run_id,
        artifact_type="strategy_source",
        path=path,
        sha256=source_hash,
    )
    PerformanceStore(workspace, store).publish(
        build_artifacts(
            run_id,
            manifest.manifest_hash,
            "test" if optimization else "test_summary" if walk_forward else "full",
            scopes,
            observations,
            selected_train_scope="trial/1/train" if optimization else None,
        )
    )
    return manifest
