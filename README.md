# TradingDev MCP Server

TradingDev 讓 LLM 透過標準 MCP 工具協助你撰寫、驗證與研究交易策略。
支援歷史回測、參數最佳化與執行結果查詢，目前不提供真實下單功能。

## 快速開始

準備 uv 與 Python 3.12 或 3.13。背景回測與最佳化需要支援 POSIX 程序群組及
`waitid(WNOWAIT)` 的系統
（macOS／Linux）；目前不支援原生 Windows 背景 worker。

在下載的專案根目錄安裝鎖定版本的依賴：

```bash
uv sync --locked
```

技術指標使用官方 `TA-Lib` Python 套件（`import talib`），由上述命令一併安裝。
支援平台的 wheel 已包含底層 C 函式庫，不需要另外執行 `brew install ta-lib`。
作業系統最低版本與 CPU 架構需求，以 uv 選用的 wheel 平台標籤為準。
若沒有相容的 wheel 而需要從原始碼編譯，請依
[TA-Lib 官方安裝說明](https://github.com/TA-Lib/ta-lib-python#installation-)準備 C 函式庫。
若要明確使用 Python 3.13，可執行 `uv sync --locked --python 3.13`。

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

從 pandas-ta 版本升級時，既有生成策略的 `import pandas_ta` 必須改用
`tradingdev.domain.indicators` 或 `talib`，再重新驗證與 dry-run。
`indicator_column` 已移除；其匯入與呼叫須改為存取指標回傳物件的具名欄位，
或直接解包 TA-Lib 的回傳 tuple，詳見[策略契約](docs/strategy_contract.md)。
SMA／EMA 週期須為 2～100000 的整數；GLFT 趨勢過濾仍可用 0 停用。
TA-Lib 的初始化、暖機期與缺值處理會改變部分指標數值，因此須重新回測、
調參與訓練 ML 模型；歷史結果與使用者工作區不會自動改寫。

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

最佳化先以 `start_optimization` 試跑並估時，等使用者同意後才呼叫
`confirm_optimization` 執行完整搜尋；以 `get_job_status`／`get_run` 查詢結果。
進入 `pending_confirmation` 後最多等待 30 分鐘；逾時工作會標示為 `failed`，
不會執行剩餘搜尋。
訓練與測試日期都包含終日，須符合 `train_start < train_end < test_start < test_end`。
呼叫指定的交易對與時間框架會覆寫 YAML；搜尋參數以外的 YAML 固定參數仍會保留。

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

報表以 `N/A` 表示缺少或沒有有限數值的指標，不代表零。

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
