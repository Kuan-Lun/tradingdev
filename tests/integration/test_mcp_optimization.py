"""Real MCP optimization: confirmation, grid search, OOS, and persisted results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import anyio
import numpy as np
import pandas as pd
import pytest
import yaml

if TYPE_CHECKING:
    from tests.integration.mcp_harness import MCPClient, MCPWorkspace

pytestmark = [pytest.mark.integration, pytest.mark.anyio]

STRATEGY_ID = "optimization_direction"
STRATEGY_CODE = '''\
"""Hold a chosen direction after a configurable warmup."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from tradingdev.domain.strategies.base import BaseStrategy

if TYPE_CHECKING:
    import pandas as pd


class DirectionStrategy(BaseStrategy):
    """Expose both searched and required fixed YAML parameters."""

    def __init__(self, direction: int, warmup: int) -> None:
        self._direction = direction
        self._warmup = warmup

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        result["signal"] = 0
        result.loc[result.index[self._warmup :], "signal"] = self._direction
        return result

    def get_parameters(self) -> dict[str, Any]:
        return {"direction": self._direction, "warmup": self._warmup}
'''


def _opposite_trends() -> pd.DataFrame:
    # Train rises for three days; OOS falls for four days. Both periods have
    # equal open/close prices, so the fee-free hold return can be hand calculated.
    prices = np.concatenate((np.arange(100.0, 172.0), np.arange(200.0, 104.0, -1)))
    return pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2024-01-01", periods=len(prices), freq="h", tz="UTC"
            ),
            "open": prices,
            "high": prices + 0.5,
            "low": prices - 0.5,
            "close": prices,
            "volume": np.full(len(prices), 1000.0),
        }
    )


async def _save_runnable(client: MCPClient) -> dict[str, Any]:
    contract = await client.call("get_strategy_contract")
    config = yaml.safe_load(contract["example_yaml_config"])
    config["strategy"].update(
        class_name="DirectionStrategy", parameters={"direction": -1, "warmup": 2}
    )
    # The MCP request must override these without editing the saved strategy.
    config["backtest"].update(symbol="ETH/USDT", timeframe="4h", fees=0, slippage=0)
    saved = await client.call(
        "save_strategy",
        strategy_id=STRATEGY_ID,
        code=STRATEGY_CODE,
        yaml_config=yaml.safe_dump(config),
    )
    assert saved["success"], saved
    for tool in ("validate_strategy", "dry_run_strategy"):
        checked = await client.call(tool, strategy_id=STRATEGY_ID)
        assert checked["success"], checked
    return dict(saved)


async def _wait_for_confirmation(client: MCPClient, job_id: str) -> dict[str, Any]:
    with anyio.fail_after(120):
        while True:
            status = await client.call("get_job_status", job_id=job_id)
            assert status["status"] not in {"failed", "done", "estimation_timeout"}, (
                json.dumps(status, indent=2)
            )
            if status["status"] == "pending_confirmation":
                return dict(status)
            await anyio.sleep(0.1)


async def test_optimization_confirmation_search_and_persisted_oos(
    mcp_workspace: MCPWorkspace,
) -> None:
    mcp_workspace.seed_market(_opposite_trends())
    async with mcp_workspace.connect() as client:
        saved = await _save_runnable(client)
        assert not (await client.call("confirm_optimization", job_id="missing"))[
            "success"
        ]
        overlapping = await client.call(
            "start_optimization",
            strategy_id=STRATEGY_ID,
            symbol="BTC/USDT",
            timeframe="1h",
            param_ranges={"direction": [-1, 0, 1]},
            optimization_metric="total_return",
            train_start="2024-01-01",
            train_end="2024-01-03",
            test_start="2024-01-03",
            test_end="2024-01-07",
        )
        assert not overlapping["job_id"]
        assert "without overlap" in overlapping["message"]
        assert await client.call("list_jobs") == []
        started = await client.call(
            "start_optimization",
            strategy_id=STRATEGY_ID,
            symbol="BTC/USDT",
            timeframe="1h",
            param_ranges={"direction": [-1, 0, 1]},
            optimization_metric="total_return",
            train_start="2024-01-01",
            train_end="2024-01-03",
            test_start="2024-01-04",
            test_end="2024-01-07",
        )
        job_id = started["job_id"]
        assert job_id and started["total_combinations"] == 3, started
        pending = await _wait_for_confirmation(client, job_id)
        assert pending["total_combinations"] == 3
        assert 1 <= pending["n_parallel_workers"] <= 2
        assert pending["time_per_combo"] >= 0
        assert pending["estimated_total_seconds"] >= 0
        # Observe beyond a worker polling interval: no result or remaining grid
        # evaluation is allowed until this client explicitly confirms.
        await anyio.sleep(2.2)
        assert (await client.call("get_job_status", job_id=job_id))[
            "status"
        ] == "pending_confirmation"
        assert await client.call("list_runs") == []
        assert not list(mcp_workspace.workspace.rglob("result.json"))
        assert (await client.call("list_jobs"))[0]["completed"] == 0

        confirmed = await client.call("confirm_optimization", job_id=job_id)
        assert confirmed["success"], confirmed
        completed = await client.wait_for_job(job_id, timeout=180)
        assert completed["status"] == "done", json.dumps(completed, indent=2)
        assert completed["best_params"] == {"direction": 1}
        assert completed["optimization_metric"] == "total_return"
        assert completed["total_combinations"] == 3
        assert (await client.call("list_jobs"))[0]["completed"] == 3
        # warmup=2 is fixed YAML, not a grid parameter. The engine enters one
        # bar later (index 3), and training must include the entire last day.
        train_return = 171.0 / 103.0 - 1
        oos_return = 105.0 / 197.0 - 1
        for metrics, expected in (
            (completed["train_metrics"], train_return),
            (completed["test_metrics"], oos_return),
        ):
            assert metrics["total_return"] == pytest.approx(expected)
            assert metrics["total_pnl"] == pytest.approx(10000 * expected)
            assert metrics["total_trades"] == 1
        assert completed["train_metrics"]["profit_factor"] is None
        assert not (await client.call("confirm_optimization", job_id=job_id))["success"]
        run = (await client.call("get_run", run_id=completed["run_id"]))["run"]
        assert run["strategy_id"] == STRATEGY_ID
        assert run["dataset_id"].startswith("BTC/USDT:1h:2024-01-01:2024-01-07:")
        result = run["metrics"]
        assert result["best_params"] == completed["best_params"]
        assert result["train_metrics"] == completed["train_metrics"]
        assert result["test_metrics"] == completed["test_metrics"]
        assert result["best_train_metric_value"] == pytest.approx(train_return)
        assert result["best_oos_metric_value"] == pytest.approx(oos_return)
        artifacts = {
            item["artifact_type"]: item
            for item in await client.call("list_artifacts", run_id=job_id)
        }
        stored = await client.call(
            "get_artifact",
            artifact_id=artifacts["result_json"]["artifact_id"],
            include_content=True,
        )
        assert "Infinity" not in stored["content"]
        assert "NaN" not in stored["content"]
        assert json.loads(stored["content"]) == result
        effective = yaml.safe_load(
            Path(artifacts["config_snapshot"]["path"]).read_text()
        )
        assert effective["backtest"]["symbol"] == "BTC/USDT"
        assert effective["backtest"]["timeframe"] == "1h"
        assert effective["backtest"]["end_date"] == "2024-01-07T23:59:59.999999"
        assert effective["data"]["requirements"]["market"]["symbol"] == "BTC/USDT"
        assert effective["strategy"]["parameters"] == {"direction": -1, "warmup": 2}
        original = yaml.safe_load(Path(saved["yaml_path"]).read_text())
        assert original["backtest"]["symbol"] == "ETH/USDT"
        assert original["backtest"]["timeframe"] == "4h"

    async with mcp_workspace.connect() as client:
        reloaded = await client.call("get_job_status", job_id=job_id)
        assert reloaded["status"] == "done"
        assert reloaded["best_params"] == {"direction": 1}
        assert reloaded["train_metrics"] == completed["train_metrics"]
        assert reloaded["test_metrics"] == completed["test_metrics"]
        assert any(item["run_id"] == job_id for item in await client.call("list_runs"))
