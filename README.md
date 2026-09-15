# TradingDev MCP Server

TradingDev 是 **MCP-first quantitative strategy development server**。主要入口
是 MCP tools；CLI 與 dashboard 是 adapters，會共用同一層 application services。
目前版本聚焦在歷史回測、策略研究、參數最佳化與 run artifact 管理；專案名稱保留
未來延伸到 paper/live execution 的空間，但現階段不提供 live trading、交易憑證
管理或下單功能。

## 快速開始

```bash
uv sync --locked
uv run python -c "import tradingdev; print('OK')"
uv run pytest
```

`pytest` 預設執行全部單元、MCP 整合及真實 Codex 端對端測試。
需有已登入的 Codex CLI 與模型網路連線；測試會自動啟動 MCP，並清除臨時產物。
CLI 會從 `PATH` 或 macOS 的 VS Code 擴充套件尋找，也可用
`TRADINGDEV_CODEX_BIN` 指定。缺少 CLI、登入或連線時，測試會回報失敗。

## 啟動 MCP

本機 stdio：

```bash
uv run tradingdev-mcp
```

可用 `--workspace /absolute/path/to/workspace` 或 `TRADINGDEV_WORKSPACE`
指定獨立工作區；背景 worker 會使用同一工作區與 Python 環境。

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
`ArtifactService`. It depends on `streamlit`/`plotly`, which live in the
`dashboard` extra rather than the base install:

```bash
uv sync --locked --extra dashboard
uv run streamlit run src/tradingdev/adapters/dashboard/app.py -- --run-id <run_id>
```

If `--run-id` is omitted, the sidebar lists runs from `workspace/tradingdev.sqlite`.

## 開發檢查

```bash
uv sync --locked --all-extras
./scripts/install-git-hooks.sh
./scripts/format.sh
./scripts/check-fast.sh
uv run --all-extras pytest
```

`uv.lock` 納入 Git，記錄各平台依賴條件與鎖定版本；重建環境時使用這份檔案。
若 `.venv` 損壞或需要重新安裝開發環境，在 macOS／Linux 的 Bash 執行：

```bash
./scripts/rebuild-env.sh
```

腳本先檢查 lockfile 是否存在且符合 `pyproject.toml`，通過才清除並重建本
repository 的 `.venv`，使用 Python 3.13 安裝鎖定的 runtime、dev 與 dashboard
依賴。它保留 lockfile、workspace 與共用 uv 快取；安裝若中斷，可重新執行。
缺少或過時的 lockfile 會在重建前失敗。修改依賴時，另外執行 `uv lock`，
檢查差異並提交 `pyproject.toml` 與 `uv.lock`；升級版本可明確使用
`uv lock --upgrade-package <package>`，完成後同步環境並執行相關測試。
跨平台 lockfile 不代表所有套件與工具都已通過 Windows 測試；目前重建與 Git
檢查腳本使用 Bash／POSIX 路徑，尚未提供原生 Windows 執行保證。

開發政策見 [AGENTS.md](AGENTS.md)。在 task branch 完成一個可驗證階段後，
以 `scripts/git-flow-commit.sh "type: message" [files...]` 提交；全部完成後，
用 `scripts/git-flow-merge.sh` 合併回主線。

Hook 安裝器也會調整本 repository 的 Git 設定：主線 merge 使用 `--no-ff`，
禁止主線 rebase，並設定 `pull.rebase=false`、`pull.ff=only`。因此後續
`git pull` 遇到分歧會停止，不會自動 merge 或 rebase；需要明確處理分歧後再整合。
安裝器遇到其他自訂 hooks 設定會停止，避免覆蓋它們。

Merge commit 只能建立在主線；把已分歧的主線 merge 進 task branch 也會被
hook 拒絕。若需要同步，可在本機 task branch 執行 `git rebase <primary>`。
整合腳本只要求兩個分支有共同祖先，不要求 task branch 先包含主線最新提交。
主線依 `tradingdev.primaryBranch`、`origin/HEAD`、唯一的 `main`／`master`
依序辨識；自訂名稱可用 `git config tradingdev.primaryBranch <branch>` 指定。

Commit hook 只做暫存內容的快速檢查。合併時會自動用 Codex 檢查程式與既有
文件是否一致，再跑完整 pytest；需要已登入的 Codex CLI 及連線。審查只讀
Git 內容，不會修改文件；過時文件、錯誤或逾時都會阻止合併，保留 task branch。
Ruff／Mypy 檢查需要 dashboard extra，以免缺少套件型別掩蓋問題。

檢查使用獨立臨時 snapshot，結束即刪除。完整檢查通過紀錄保存在 Git metadata，
相同內容與檢查環境可重用；主線 push 也必須有通過紀錄。
可在已提交且乾淨的分支執行 `uv run --no-sync python scripts/git_gate.py full`
預先檢查。Codex 文件審查是語意輔助，仍可能漏判。

## 相關文件

- [ARCHITECTURE.md](ARCHITECTURE.md)
- [docs/strategy_contract.md](docs/strategy_contract.md)
- [docs/run_artifacts.md](docs/run_artifacts.md)
- [docs/strategies/](docs/strategies/)
