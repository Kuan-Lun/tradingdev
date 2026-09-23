# 開發指南

所有命令均在 repository 根目錄執行。產品操作見 [README](../README.md)，
代理開發政策見 [AGENTS.md](../AGENTS.md)。

## 環境與常用命令

首次設置：

```bash
uv sync --locked --all-extras
./scripts/install-git-hooks.sh
```

安裝器會調整本機 Git 設定，摘要見文末的[注意事項](#注意事項)。

| 命令 | 用途 |
| --- | --- |
| `./scripts/rebuild-env.sh` | 重建 `.venv`，依 `uv.lock` 安裝鎖定版本。 |
| `./scripts/format.sh` | 自動修正 lint 與格式，會修改檔案。 |
| `./scripts/check-fast.sh` | 唯讀檢查 Ruff、格式、strict Mypy 與 Markdown。 |
| `./scripts/check-full.sh` | 執行上述品質檢查與完整離線 pytest；不呼叫模型。 |
| `uv run --all-extras pytest` | 完整離線測試，包含真實 MCP／worker；不呼叫模型。 |

## 選擇性 LLM 測試

兩個入口共用均線、動量、錯誤草稿修復與舊格式策略恢復情境，要求模型完成策略生成、回測及
結果查詢；腳本獨立核對訊號與結果，並清理臨時檔案及程序。
入口即時顯示進度與耗時，第一個失敗就停止並顯示原因；要跑完全部情境可加
`--maxfail=0`。

```bash
./scripts/check-llm.sh codex --llm-model gpt-5.6-luna
./scripts/check-llm.sh local --llm-model qwen3.8:27b \
  --llm-reasoning-effort none --llm-temperature 0.7 --llm-timeout 900
```

本地測試已驗證支援 Ollama 的 Qwen3.8 27B。模型可用 `ollama pull qwen3.8:27b` 安裝。
Local 使用支援工具呼叫的 Chat Completions 服務，預設
`http://localhost:11434/v1`；其他本機服務以 `--llm-base-url` 指定。
其他支援工具呼叫的本地模型可用 `--llm-model` 指定，須另行執行測試驗證。
未指定 temperature 時採服務 API 的預設值，不保證等於模型設定檔；需要覆寫時
加 `--llm-temperature 1.0`（範圍 0–2）。
推理強度以 `--llm-reasoning-effort` 指定，支援值依模型服務而定；省略時使用服務預設。
快速檢查可加 `-k sma`，只跑均線的生成、回測與查詢；移除此選項才涵蓋全部四個情境。
`-k legacy` 驗證模型先探索及讀取舊格式策略，再另存、驗證新 revision 並執行回測，
同時確認原有檔案未變。
`--llm-timeout 900` 是每個模型情境的秒數上限，不是預期執行時間。

## 從開發到合併

以下以 remote `origin`、主線 `main`、開發分支 `feature/my-task` 示範，
請使用你實際的名稱。

### 1. 分階段提交並推送

在開發分支完成一個階段後，先 `git add` 要提交的檔案，再填寫提交訊息：

```bash
./scripts/git-flow-commit.sh "feat: add strategy validation"
git push -u origin feature/my-task
```

Commit 會自動跑快速檢查；一般 push 不執行完整測試或呼叫模型。

### 2. 在 GitHub 建立 PR，交由審閱者檢查

從開發分支向主線建立 PR。編號取自 PR 標題的 `#編號` 或網址 `/pull/編號`；
例如 `/pull/27` 的編號就是 `27`。以下 `27` 均為範例，請換成實際 PR 編號：

```bash
./scripts/check-pr.sh 27
```

審閱者可留在目前開發分支執行；腳本會自動檢查合併候選、執行 Codex 文件
審查與完整離線 pytest，並清理臨時產物；真實模型策略測試由審閱者視變更另行執行。

### 3. 在 GitHub 審閱並合併

完成評論與審閱後，核對下列兩個版本是否仍與檢查結果相同，再於網頁合併：

```bash
git ls-remote origin refs/heads/main refs/pull/27/head
```

### 4. 合併後更新 main、清理本機分支

```bash
git switch main
git pull --ff-only origin main
./scripts/cleanup-pr.sh 27 --branch feature/my-task
```

腳本核對合併狀態後，只刪除符合條件的本機分支；無法確認時會保留並說明原因。

## 尚未建立 PR 的本機檢查

```bash
uv run --no-sync python scripts/git_gate.py full --base main
```

此命令檢查指定本機主線與目前提交的合併候選。

## 注意事項

- **測試清理**：若無法確認 worker 已停止，測試會報錯並列出保留的臨時目錄，
  供確認程序後清理。監督程序需具備程序群組觀察權限；若被外部強制殺死，
  不會假定其子程序已一併停止。
- **Git 設定**：安裝器只調整本 repository，設定 `core.hooksPath=.githooks`、
  `branch.<主線>.rebase=false` 與 `pull.ff=only`，讓一般 pull 遇分歧時停止。
  遷移時僅移除唯一值為 `--no-ff` 的 `branch.<主線>.mergeOptions`，
  保留其他自訂 merge options 與 `pull.rebase`。
- **環境**：腳本使用 Bash，目前未保證原生 Windows 相容；Git 需支援
  `merge-tree --write-tree`。更新依賴時執行 `uv lock`，並提交 lockfile 與設定變更。
- **模型與費用**：日常 pytest 不需模型服務。Codex 策略測試及 PR 文件審查
  需要已登入的 Codex CLI、網路與額度；可用 `TRADINGDEV_CODEX_BIN` 指定 CLI。
  上方 Codex 範例明確指定 Luna，須有該模型的存取權限；可用 `--llm-model`
  指定其他模型。模型不可用時測試會失敗，不會自動改用其他模型。
  本地測試須自行安裝並啟動模型服務，不會自動下載模型或改用付費服務；本地
  模型通過不代表 Codex／Claude 相容性已驗證。PR 檢查與清理另需 `gh` 能存取 repository。
- **PR 與 remote**：PR 檢查要求乾淨且已提交的工作樹，以及開啟且指向主線的 PR。
  兩個 PR 腳本僅支援 `github.com` 的標準 HTTPS／SSH remote，預設為 `origin`；
  例如改用名為 `upstream` 的 remote，就加 `--remote upstream`。
  自訂主線使用 `git config tradingdev.primaryBranch 分支名稱`。
- **合併前版本**：主線或 PR 有新提交就重新檢查。本機檢查不能攔截 GitHub 網頁
  合併；尚未建立 PR 時的本機檢查，也不代表遠端 PR 已驗證。
- **分支清理**：先在 GitHub 合併再清理；squash／rebase 合併可能需要人工確認。
  遠端分支與 worktree 會保留。
- **舊版遷移**：重新安裝 Git hooks；若客戶端仍註冊指向已刪除的
  `scripts/hooks/finalize-*` 或 `.Codex/hooks/` 的 Stop hooks，請移除這些註冊。
