# TradingDev MCP Server

TradingDev 讓 LLM 透過標準 MCP 工具協助你撰寫、驗證與研究交易策略。
支援歷史回測、參數最佳化與執行結果查詢，目前不提供真實下單功能。

## 快速開始

準備 uv 與 Python 3.12 或 3.13，在下載的專案根目錄安裝鎖定版本的依賴：

```bash
uv sync --locked
```

接著依下一節啟動 MCP server，並將它加入你使用的 MCP 客戶端。

## 啟動 MCP

本機 stdio：

```bash
uv run tradingdev-mcp
```

可用 `--workspace /absolute/path/to/workspace` 或 `TRADINGDEV_WORKSPACE`
指定存放生成策略、資料及執行結果的工作區。

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

## 策略開發流程

1. `list_strategies`：先檢查 bundled/generated strategy。
2. `get_strategy_contract`：取得 LLM 產生策略必須遵守的 Python/YAML 契約。
3. `save_strategy`：只把 generated strategy 存成 draft。
4. `validate_strategy`：跑 syntax、static policy、ruff、mypy、繼承與 signal
   contract 檢查。
5. `dry_run_strategy`：只接受 validated strategy，通過後升為 runnable。
6. `start_backtest` 或 `start_walk_forward`：只接受 runnable/promoted strategy。
7. `get_job_status`、`list_runs`、`compare_runs`、`list_artifacts` 查詢結果。

`inspect_dataset(config_path)` 可在執行前檢查策略需要的行情、特徵資料、
檔案位置與缺值狀態。

## MCP 工具

| 類別 | Tools |
| ---- | ----- |
| Strategy | `get_strategy_contract`, `list_strategies`, `get_strategy`, `save_strategy`, `validate_strategy`, `dry_run_strategy` |
| Data | `list_available_data`, `inspect_dataset`, `ensure_data` |
| Backtest | `start_backtest`, `start_walk_forward` |
| Optimization | `start_optimization`, `confirm_optimization` |
| Jobs/Runs | `get_job_status`, `list_jobs`, `cancel_job`, `list_runs`, `get_run`, `compare_runs` |
| Artifacts | `list_artifacts`, `get_artifact`, `promote_strategy` |
| Requests | `record_feature_request`, `list_feature_requests` |

## 工作區與檔案

- 內建策略：
  `src/tradingdev/domain/strategies/bundled/<strategy>/strategy.py`
- 內建策略設定：
  `src/tradingdev/domain/strategies/bundled/<strategy>/config.yaml`
- 透過 MCP 產生的策略：
  `workspace/generated_strategies/<strategy_id>.py`
- 生成策略的設定：
  `workspace/configs/<strategy_id>.yaml`
- 行情資料快取：
  `workspace/data/raw/` 與 `workspace/data/processed/`
- 工作狀態與執行結果索引：
  `workspace/tradingdev.sqlite`
- 每次執行的結果檔案：
  `workspace/runs/<run_id>/`，包含結果、使用的設定與策略副本、資料集識別資訊
  及 dashboard 所需檔案。

`workspace/` 是預設工作區；指定其他 `--workspace` 路徑時，生成策略、
資料與執行結果會存到該路徑。內建策略與設定隨專案提供，透過 MCP 撰寫的
策略另存於工作區。

## CLI

也可以在終端機直接執行策略設定檔：

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

Dashboard 可查看已完成的執行結果。使用前需安裝 `dashboard` 額外依賴：

```bash
uv sync --locked --extra dashboard
uv run streamlit run src/tradingdev/adapters/dashboard/app.py -- --run-id <run_id>
```

`<run_id>` 是要查看的執行結果識別碼，可透過 MCP 的 `list_runs` 取得。
省略 `--run-id` 時，可從側邊欄選擇工作區中已儲存的執行結果。

## 相關文件

- [策略撰寫契約](docs/strategy_contract.md)
- [執行結果與產物](docs/run_artifacts.md)
- [內建策略說明](docs/strategies/)
- [開發指南](docs/development.md)：開發環境重建、品質檢查、測試、Git／PR 流程與分支清理。
- [專案架構](ARCHITECTURE.md)
- [代理開發政策](AGENTS.md)
