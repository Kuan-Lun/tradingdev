"""Exercise the client-visible strategy workflow over real stdio MCP."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
import yaml

if TYPE_CHECKING:
    import pandas as pd

    from tests.integration.mcp_harness import MCPClient, MCPWorkspace

pytestmark = [pytest.mark.integration, pytest.mark.anyio]

STRATEGY_ID = "prompt_sma_strategy"
PROMPT = "建立 BTC/USDT 一小時均線交叉策略，快線 3、慢線 8，完成驗證與回測。"
RUN_ARGUMENTS = {
    "strategy_id": STRATEGY_ID,
    "symbol": "BTC/USDT",
    "timeframe": "1h",
    "start_date": "2024-01-01",
    "end_date": "2024-01-08",
}


async def save_example(client: MCPClient) -> tuple[str, str, dict[str, Any]]:
    """Replay deterministic LLM output using the advertised contract."""
    contract = await client.call("get_strategy_contract")
    code = contract["example_strategy_code"]
    # Required YAML parameters catch workers silently using constructor defaults.
    code = code.replace(
        "        backtest_engine: BaseBacktestEngine | None = None,\n", ""
    )
    code = code.replace("fast_period: int = 10", "fast_period: int")
    code = code.replace(
        "slow_period: int = 30,",
        "slow_period: int,\n        backtest_engine: BaseBacktestEngine | None = None,",
    )
    config = yaml.safe_load(contract["example_yaml_config"])
    config["strategy"]["parameters"] = {"fast_period": 3, "slow_period": 8}
    config["backtest"]["symbol"] = "ETH/USDT"
    config["backtest"]["random_seed"] = 42
    config_text = yaml.safe_dump(config)
    saved = await client.call(
        "save_strategy",
        strategy_id=STRATEGY_ID,
        code=code,
        yaml_config=config_text,
        request_summary=PROMPT,
    )
    assert saved["success"], saved
    assert saved["status"] == "draft"
    return code, config_text, dict(saved)


async def make_runnable(client: MCPClient, revision_id: str | None = None) -> None:
    validated = await client.call(
        "validate_strategy", strategy_id=STRATEGY_ID, revision_id=revision_id
    )
    assert validated["success"], json.dumps(validated, indent=2)
    assert validated["status"] == "validated"
    dry_run = await client.call(
        "dry_run_strategy", strategy_id=STRATEGY_ID, revision_id=revision_id
    )
    assert dry_run["success"], json.dumps(dry_run, indent=2)
    assert dry_run["status"] == "runnable"
    assert dry_run["signal_analysis"]["rows"] == 240
    assert dry_run["signal_analysis"]["nan_count"] == 0


async def test_generated_strategy_full_mcp_workflow(
    mcp_workspace: MCPWorkspace,
    sample_ohlcv_with_kd: pd.DataFrame,
) -> None:
    cache_path = mcp_workspace.seed_market(sample_ohlcv_with_kd)
    async with mcp_workspace.connect() as client:
        tools = {tool.name: tool for tool in (await client.session.list_tools()).tools}
        assert {
            "get_strategy_contract",
            "save_strategy",
            "validate_strategy",
            "dry_run_strategy",
            "start_backtest",
            "get_job_status",
            "get_artifact",
        } <= tools.keys()
        assert {"strategy_id", "code", "yaml_config"} <= set(
            tools["save_strategy"].inputSchema["required"]
        )
        assert all(
            item["kind"] == "bundled" for item in await client.call("list_strategies")
        )
        code, config_text, saved = await save_example(client)
        for key in ("py_path", "yaml_path"):
            assert Path(saved[key]).is_relative_to(mcp_workspace.workspace)
        assert Path(saved["py_path"]).read_text(encoding="utf-8") == code
        draft = await client.call("get_strategy", strategy_id=STRATEGY_ID)
        assert draft["metadata"]["request_summary"] == PROMPT
        assert draft["source_code"] == code
        assert not (await client.call("dry_run_strategy", strategy_id=STRATEGY_ID))[
            "success"
        ]
        assert not (await client.call("promote_strategy", strategy_id=STRATEGY_ID))[
            "success"
        ]
        assert not (await client.call("start_backtest", **RUN_ARGUMENTS))["job_id"]
        assert await client.call("list_jobs") == []
        await make_runnable(client, saved["revision_id"])
        promoted = await client.call(
            "promote_strategy",
            strategy_id=STRATEGY_ID,
            revision_id=saved["revision_id"],
        )
        assert promoted["success"] and promoted["status"] == "promoted"
        assert "binance_vision" in await client.call("list_data_sources")
        data = await client.call(
            "ensure_data",
            **{k: v for k, v in RUN_ARGUMENTS.items() if k != "strategy_id"},
        )
        assert data["success"] and data["rows"] > 0
        assert data["processed_path"] == str(cache_path)
        started = await client.call(
            "start_backtest", **RUN_ARGUMENTS, revision_id=saved["revision_id"]
        )
        assert started["revision_id"] == saved["revision_id"]
        assert len(started["manifest_hash"]) == 64
        assert started["job_id"] and started["data_available"], started
        completed = await client.wait_for_job(started["job_id"])
        assert completed["status"] == "done", completed
        assert completed["revision_id"] == saved["revision_id"]
        assert completed["manifest_hash"] == started["manifest_hash"]
        assert completed["metrics"]["total_trades"] > 0
        run_id = completed["run_id"]
        run = (await client.call("get_run", run_id=run_id))["run"]
        assert run["strategy_id"] == STRATEGY_ID
        assert run["revision_id"] == saved["revision_id"]
        assert run["manifest_hash"] == started["manifest_hash"]
        assert run["metrics"] == completed["metrics"]
        artifacts = await client.call("list_artifacts", run_id=run_id)
        by_type = {item["artifact_type"]: item for item in artifacts}
        assert {
            "result_json",
            "config_snapshot",
            "strategy_source",
            "dataset_fingerprint",
            "pipeline_result",
            "execution_manifest",
        } <= by_type.keys()
        for artifact in artifacts:
            path = Path(artifact["path"])
            assert path.is_relative_to(mcp_workspace.workspace / "runs")
            assert hashlib.sha256(path.read_bytes()).hexdigest() == artifact["sha256"]
        source = await client.call(
            "get_artifact",
            artifact_id=by_type["strategy_source"]["artifact_id"],
            include_content=True,
        )
        assert source["content"] == code
        config_artifact = await client.call(
            "get_artifact",
            artifact_id=by_type["config_snapshot"]["artifact_id"],
            include_content=True,
        )
        effective = yaml.safe_load(config_artifact["content"])
        assert effective["strategy"]["revision_id"] == saved["revision_id"]
        assert effective["strategy"]["parameters"] == {
            "fast_period": 3,
            "slow_period": 8,
        }
        assert effective["backtest"]["symbol"] == "BTC/USDT"
        assert effective["backtest"]["end_date"].startswith("2024-01-08")
        assert effective["data"]["requirements"]["market"]["symbol"] == "BTC/USDT"
        manifest_artifact = await client.call(
            "get_artifact",
            artifact_id=by_type["execution_manifest"]["artifact_id"],
            include_content=True,
        )
        manifest = json.loads(manifest_artifact["content"])
        assert manifest["schema_version"] == 2
        assert manifest["strategy_execution"]["constructor_kwargs"] == {
            "fast_period": 3,
            "slow_period": 8,
        }
        assert manifest["kind"] == "backtest"
        assert manifest["manifest_hash"] == started["manifest_hash"]
        assert manifest["config"] == effective
        assert (
            yaml.safe_load(Path(saved["yaml_path"]).read_text())["backtest"]["symbol"]
            == "ETH/USDT"
        )
        inspection = await client.call(
            "inspect_dataset", config_path=by_type["config_snapshot"]["path"]
        )
        assert inspection["data_root"] == str(mcp_workspace.data_root)
        assert inspection["market_available"]
        assert inspection["market"]["rows"] == len(sample_ohlcv_with_kd)

    # Persist across MCP sessions; a new draft does not inherit old approval.
    async with mcp_workspace.connect() as client:
        assert (await client.call("get_job_status", job_id=started["job_id"]))[
            "status"
        ] == "done"
        assert any(item["run_id"] == run_id for item in await client.call("list_runs"))
        assert (await client.call("get_strategy", strategy_id=STRATEGY_ID))["metadata"][
            "status"
        ] == "promoted"
        revision = await client.call(
            "save_strategy",
            strategy_id=STRATEGY_ID,
            code=code + "\n# Revised draft\n",
            yaml_config=config_text,
        )
        assert revision["status"] == "draft"
        assert revision["revision_id"] != saved["revision_id"]
        old = await client.call(
            "get_strategy", strategy_id=STRATEGY_ID, revision_id=saved["revision_id"]
        )
        assert old["metadata"]["status"] == "promoted"
        assert old["source_code"] == code
        rerun = await client.call(
            "start_backtest", **RUN_ARGUMENTS, revision_id=saved["revision_id"]
        )
        rerun_done = await client.wait_for_job(rerun["job_id"])
        assert rerun_done["status"] == "done"
        assert rerun_done["revision_id"] == saved["revision_id"]
        assert rerun_done["metrics"] == completed["metrics"]
        assert not (await client.call("start_backtest", **RUN_ARGUMENTS))["job_id"]
        assert Path(by_type["strategy_source"]["path"]).read_text() == code


async def test_rejected_drafts_diagnostics_and_repair(
    mcp_workspace: MCPWorkspace,
) -> None:
    async with mcp_workspace.connect() as client:
        code, config_text, saved = await save_example(client)
        before = {
            path: path.read_bytes()
            for path in mcp_workspace.workspace.rglob("*")
            if path.is_file()
        }
        for args in (
            {"strategy_id": "../escaped", "code": code, "yaml_config": config_text},
            {
                "strategy_id": STRATEGY_ID,
                "code": "def broken(:",
                "yaml_config": config_text,
            },
            {"strategy_id": STRATEGY_ID, "code": code, "yaml_config": "strategy: ["},
        ):
            assert not (await client.call("save_strategy", **args))["success"]
        assert {path: path.read_bytes() for path in before} == before
        assert not (mcp_workspace.root / "escaped.py").exists()
        marker = mcp_workspace.root / "must-not-execute.txt"
        forbidden = (
            f"from pathlib import Path\nPath({str(marker)!r}).write_text('executed')\n"
        )
        await client.call(
            "save_strategy",
            strategy_id=STRATEGY_ID,
            code=forbidden,
            yaml_config=config_text,
        )
        rejected = await client.call("validate_strategy", strategy_id=STRATEGY_ID)
        assert not rejected["success"] and rejected["status"] == "draft"
        assert "banned_import" in {item["code"] for item in rejected["diagnostics"]}
        assert not marker.exists(), "Rejected code must never be imported"
        removed_import = code.replace(
            "from __future__ import annotations",
            "from __future__ import annotations\n\nimport pandas_ta",
        )
        await client.call(
            "save_strategy",
            strategy_id=STRATEGY_ID,
            code=removed_import,
            yaml_config=config_text,
        )
        rejected = await client.call("validate_strategy", strategy_id=STRATEGY_ID)
        assert not rejected["success"] and rejected["status"] == "draft"
        import_diagnostic = next(
            item
            for item in rejected["diagnostics"]
            if item["code"] == "import_not_allowed"
        )
        assert import_diagnostic["message"] == "import not allowed: pandas_ta"
        assert "talib" in import_diagnostic["fix"]
        for root in (
            "__future__",
            "collections",
            "dataclasses",
            "enum",
            "statistics",
            "typing_extensions",
        ):
            assert root in import_diagnostic["fix"]
        bad_signal = code.replace('result["signal"] = 0', 'result["signal"] = 7')
        await client.call(
            "save_strategy",
            strategy_id=STRATEGY_ID,
            code=bad_signal,
            yaml_config=config_text,
        )
        rejected = await client.call("validate_strategy", strategy_id=STRATEGY_ID)
        assert not rejected["success"]
        assert "invalid_signal_values" in {
            item["code"] for item in rejected["diagnostics"]
        }
        assert all(item.get("fix") for item in rejected["diagnostics"])
        repaired = await client.call(
            "save_strategy", strategy_id=STRATEGY_ID, code=code, yaml_config=config_text
        )
        await make_runnable(client, repaired["revision_id"])
        metadata_path = Path(repaired["py_path"]).parent / "metadata.json"
        assert json.loads(metadata_path.read_text())["status"] == "runnable"
        assert (
            json.loads((Path(saved["py_path"]).parent / "metadata.json").read_text())[
                "status"
            ]
            == "draft"
        )


async def test_worker_failure_is_reported_and_cancelled_job_stops(
    mcp_workspace: MCPWorkspace,
    sample_ohlcv_with_kd: pd.DataFrame,
) -> None:
    cache = mcp_workspace.seed_market(sample_ohlcv_with_kd)
    async with mcp_workspace.connect() as client:
        await save_example(client)
        await make_runnable(client)
        cache.write_bytes(b"invalid parquet")
        started = await client.call("start_backtest", **RUN_ARGUMENTS)
        failed = await client.wait_for_job(started["job_id"])
        assert failed["status"] == "failed", failed
        assert failed["error"]
        assert await client.call("list_runs") == []
        mcp_workspace.seed_market(sample_ohlcv_with_kd)
        started = await client.call("start_backtest", **RUN_ARGUMENTS)
        cancelled = await client.call("cancel_job", job_id=started["job_id"])
        assert cancelled["success"] and cancelled["process_terminated"], cancelled
        assert (await client.call("get_job_status", job_id=started["job_id"]))[
            "status"
        ] == "cancelled"
        assert not (await client.call("cancel_job", job_id=started["job_id"]))[
            "success"
        ]


async def test_real_quality_gates_reject_invalid_llm_output(
    mcp_workspace: MCPWorkspace,
) -> None:
    async with mcp_workspace.connect() as client:
        code, config_text, _ = await save_example(client)
        await client.call(
            "save_strategy",
            strategy_id=STRATEGY_ID,
            code=code + "\nUNKNOWN = undefined_variable\n",
            yaml_config=config_text,
        )
        rejected = await client.call("validate_strategy", strategy_id=STRATEGY_ID)
        assert not rejected["success"]
        codes = {item["code"] for item in rejected["diagnostics"]}
        assert {"ruff_failed", "mypy_failed"} <= codes
        assert "contract_execution_error" not in codes
        assert not (await client.call("start_backtest", **RUN_ARGUMENTS))["job_id"]
        malformed = await client.session.call_tool(
            "save_strategy", {"strategy_id": STRATEGY_ID}
        )
        assert malformed.isError
        assert (await client.call("get_strategy", strategy_id=STRATEGY_ID))["metadata"][
            "status"
        ] == "draft"


async def test_walk_forward_runs_through_real_worker(
    mcp_workspace: MCPWorkspace,
    sample_ohlcv_with_kd: pd.DataFrame,
) -> None:
    mcp_workspace.seed_market(sample_ohlcv_with_kd)
    async with mcp_workspace.connect() as client:
        code, config_text, _ = await save_example(client)
        config = yaml.safe_load(config_text)
        config["validation"] = {"n_splits": 2, "train_ratio": 0.6}
        await client.call(
            "save_strategy",
            strategy_id=STRATEGY_ID,
            code=code,
            yaml_config=yaml.safe_dump(config),
        )
        await make_runnable(client)
        wrong_mode = await client.call("start_backtest", **RUN_ARGUMENTS)
        assert not wrong_mode["job_id"]
        assert "start_walk_forward" in wrong_mode["message"]
        started = await client.call("start_walk_forward", **RUN_ARGUMENTS)
        assert started["job_id"], started
        completed = await client.wait_for_job(started["job_id"])
        assert completed["status"] == "done", completed
        assert completed["metrics"]["n_folds"] == 2
        artifacts = await client.call("list_artifacts", run_id=completed["run_id"])
        assert any(item["artifact_type"] == "pipeline_result" for item in artifacts)
