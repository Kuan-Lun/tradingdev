# TradingDev MCP Server

TradingDev 是 **MCP-first quantitative strategy development server**。主要入口
是 MCP tools；CLI 與 dashboard 是 adapters，會共用同一層 application services。
目前版本聚焦在歷史回測、策略研究、參數最佳化與 run artifact 管理；專案名稱保留
未來延伸到 paper/live execution 的空間，但現階段不提供 live trading、交易憑證
管理或下單功能。

## 快速開始

```bash
uv sync
uv run python -c "import tradingdev; print('OK')"
uv run pytest tests/
```

## 啟動 MCP

本機 stdio：

```bash
uv run tradingdev-mcp
```

HTTP / streamable-http：

```bash
uv run tradingdev-mcp --web --transport streamable-http --port 8000
```

Claude Desktop 範例：

```json
{
  "mcpServers": {
    "tradingdev": {
      "command": "uv",
      "args": ["run", "tradingdev-mcp"],
      "cwd": "/absolute/path/to/tradingdev.clone"
    }
  }
}
```

## MCP Workflow

1. `list_strategies`：先檢查 bundled/generated strategy。
2. `get_strategy_contract`：取得 LLM 產生策略必須遵守的 Python/YAML 契約。
3. `save_strategy`：只把 generated strategy 存成 draft。
4. `validate_strategy`：跑 syntax、static policy、ruff、mypy、繼承與 signal
   contract 檢查。
5. `dry_run_strategy`：只接受 validated strategy，通過後升為 runnable。
6. `start_backtest` 或 `start_walk_forward`：只接受 runnable/promoted strategy。
7. `get_job_status`、`list_runs`、`compare_runs`、`list_artifacts` 查詢結果。

`inspect_dataset(config_path)` can be used before a run to inspect declared
market and feature requirements, feature paths, and missing-value status.

## MCP Tools

| 類別 | Tools |
| ---- | ----- |
| Strategy | `get_strategy_contract`, `list_strategies`, `get_strategy`, `save_strategy`, `validate_strategy`, `dry_run_strategy` |
| Data | `list_available_data`, `inspect_dataset`, `ensure_data` |
| Backtest | `start_backtest`, `start_walk_forward` |
| Optimization | `start_optimization`, `confirm_optimization` |
| Jobs/Runs | `get_job_status`, `list_jobs`, `cancel_job`, `list_runs`, `get_run`, `compare_runs` |
| Artifacts | `list_artifacts`, `get_artifact`, `promote_strategy` |
| Requests | `record_feature_request`, `list_feature_requests` |

## 路徑與資料模型

- Bundled strategies:
  `src/tradingdev/domain/strategies/bundled/<strategy>/strategy.py`
- Bundled configs:
  `src/tradingdev/domain/strategies/bundled/<strategy>/config.yaml`
- Generated strategies:
  `workspace/generated_strategies/<strategy_id>.py`
- Generated configs:
  `workspace/configs/<strategy_id>.yaml`
- Runtime data cache:
  `workspace/data/raw/` 與 `workspace/data/processed/`
- Job/run/artifact metadata:
  `workspace/tradingdev.sqlite`
- Run files:
  `workspace/runs/<run_id>/` with result, config snapshot, strategy source
  snapshot, dataset fingerprint, and dashboard pipeline artifacts.

`workspace/` 是 runtime 工作區，不進 wheel。MCP 只能寫入 workspace；git 版控的
bundled strategy/config 由工程師維護。

## CLI

CLI 是 adapter，適合本機工程師直接執行 config：

```bash
uv run python -m tradingdev --config \
  src/tradingdev/domain/strategies/bundled/kd_strategy/config.yaml
```

含 `validation:` 的 config 需明確執行 walk-forward：

```bash
uv run python -m tradingdev --config \
  src/tradingdev/domain/strategies/bundled/xgboost_strategy/config.yaml \
  --walk-forward
```

## Dashboard

The dashboard reads completed MCP runs through `RunService` and
`ArtifactService`:

```bash
uv run streamlit run src/tradingdev/adapters/dashboard/app.py -- --run-id <run_id>
```

If `--run-id` is omitted, the sidebar lists runs from `workspace/tradingdev.sqlite`.

## 開發檢查

```bash
./scripts/hooks/finalize-python.sh
./scripts/hooks/finalize-markdown.sh
uv run pytest tests/
```

## 相關文件

- [ARCHITECTURE.md](ARCHITECTURE.md)
- [docs/strategy_contract.md](docs/strategy_contract.md)
- [docs/run_artifacts.md](docs/run_artifacts.md)
- [docs/strategies/](docs/strategies/)
