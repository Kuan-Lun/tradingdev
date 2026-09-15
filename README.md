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
repository 的 `.venv`，由 uv 依 `pyproject.toml` 選擇相容的 Python（目前為
3.12 或 3.13），安裝鎖定的 runtime、dev 與 dashboard 依賴。它保留 lockfile、
workspace 與共用 uv 快取；安裝若中斷，可重新執行。
缺少或過時的 lockfile 會在重建前失敗。修改依賴時，另外執行 `uv lock`，
檢查差異並提交 `pyproject.toml` 與 `uv.lock`；升級版本可明確使用
`uv lock --upgrade-package <package>`，完成後同步環境並執行相關測試。
跨平台 lockfile 不代表所有套件與工具都已通過 Windows 測試；目前重建與 Git
檢查腳本使用 Bash／POSIX 路徑，尚未提供原生 Windows 執行保證。

開發政策見 [AGENTS.md](AGENTS.md)。在 task branch 完成一個可驗證階段後，
以 `scripts/git-flow-commit.sh "type: message" [files...]` 提交。一般 push 不跑
完整測試或呼叫模型；PR 評論、審閱與合併在 GitHub 網頁進行。工作完成時
保留分支，不自動在本機合併主線；舊 `scripts/git-flow-merge.sh` 已移除。

由舊版遷移時，請從 `.claude/settings.local.json` 等客戶端設定移除指向
`scripts/hooks/finalize-python.sh`、`scripts/hooks/finalize-markdown.sh` 或
`.Codex/hooks/` 下對應 wrapper 的 Stop hook 註冊；這些腳本已移除。
Git hook 安裝器不修改客戶端設定。手動格式化與快速檢查使用上面的
`scripts/format.sh`、`scripts/check-fast.sh`。

重新執行 `scripts/install-git-hooks.sh` 可遷移既有設定：僅當本機主線
`mergeOptions` 唯一值為舊版 `--no-ff` 時移除它，保留其他自訂值；設定
主線 `rebase=false`、`pull.ff=only`，保留開發者的 `pull.rebase`。
一般 pull 遇到分歧會停止；可明確 merge 主線到 task branch，不強制對共用
分支 rebase。安裝器遇到其他自訂 hooks 設定會停止，避免覆蓋它們。
主線依 `tradingdev.primaryBranch`、`origin/HEAD`、本機或 origin 唯一的
`main`／`master` 依序辨識；自訂名稱可用
`git config tradingdev.primaryBranch <branch>` 指定，不要求存在本機主線。

Commit hook 驗證暫存內容的獨立 snapshot，包含 task branch 合併與解決衝突
後的提交。主線直接 commit、建立 merge commit、rebase、push（含刪除）
會被拒絕。Git hooks 無法攔截 GitHub 網頁合併，也不攔截本機 fast-forward
參照更新；主線用來追蹤已在 GitHub 合併的結果。

### 審閱者在本機檢查 PR

負責決定合併的人，在已提交且乾淨的 checkout 明確執行下列命令。
需要本機 `gh` 已登入且能讀取 repository，以及已登入的 Codex CLI、連線
及本專案完整開發環境；不需要雲端 Codex CI 或把憑證交給別人。

```bash
# 將 123 換成待審查的 PR 編號；可加 --remote upstream 指定目標 remote。
./scripts/check-pr.sh 123
```

腳本讀取指定 PR，在獨立臨時 repository fetch 當前主線與 PR head，計算
合併候選內容。版本不一致或合併衝突會立即失敗；它不切換目前分支。
接著執行 Codex 程式／文件一致性審查及 `scripts/check-full.sh`，包含快速
檢查和完整 pytest（真實 Codex 經 MCP 撰寫策略的測試也在內）。Ruff／Mypy
需要 dashboard extra，以免缺少套件型別掩蓋問題。

文件審查只讀 Git 內容，不修改文件；過時文件、錯誤或逾時會令命令失敗。
每次執行都重新檢查，不使用舊版完整檢查快取。臨時 repository、snapshot
與測試產物在結束時清理；快照執行上限 900 秒，文件審查上限 180 秒，
程序清理時間另計。Codex 語意審查仍可能漏判。

結束時再次核對 PR，輸出 URL、base／head SHA、候選 tree 與結果，供審閱者
自行貼到 PR。腳本不發送評論或合併。網頁合併前必須核對目前 base／head；
任一改變便重新檢查。可在自己的終端機讀取目前版本：

```bash
gh pr view 123 --json url,state,baseRefOid,headRefOid
```

純本機檢查無法知道使用者何時按下 GitHub 合併，也不能阻止核對後的版本
變更；目前靠審閱者遵守此流程，沒有遠端強制檢查門禁。

尚未建立 PR 時，可檢查本機指定主線與目前提交的合併候選：

```bash
uv run --no-sync python scripts/git_gate.py full --base main
```

這不驗證遠端 PR；`--base` 必須明確指定，也可加 `--head <ref>`。

### PR 合併後清理本機分支

在 GitHub 合併後，切換到其他分支，再明確指定已完成的 PR 與本機分支：

```bash
./scripts/cleanup-pr.sh 123 --branch feature/my-task
```

可加 `--remote upstream` 指定 PR 目標 repository。腳本只 fetch 目標主線，
不執行 pull。它核對 PR 已合併、本機 upstream repository／分支與 PR
來源相符、本機 tip 等於 PR head，以及 PR head／merge commit 都已在
取得的主線歷史中，才原子刪除指定的本機分支 ref。檢查後若分支增加提交，
刪除會失敗。它不依靠 commit message 的分支名稱，也不呼叫 LLM。

主線與 `git config --add tradingdev.protectedBranch <branch>` 指定的分支
會保留；正在任何 worktree checkout 的分支也不刪除。若 GitHub 使用
squash／rebase 而無法證明原始 head 已包含於主線，腳本保留分支並要求人工
確認；fetch 不會恢復改寫過的祖先關係。

遠端分支、worktree 與本機 branch config 都保留。保留 config 是為了避免
ref 刪除後與同名新分支競態而誤刪設定；命令結果會明確列出此行為。

## 相關文件

- [ARCHITECTURE.md](ARCHITECTURE.md)
- [docs/strategy_contract.md](docs/strategy_contract.md)
- [docs/run_artifacts.md](docs/run_artifacts.md)
- [docs/strategies/](docs/strategies/)
