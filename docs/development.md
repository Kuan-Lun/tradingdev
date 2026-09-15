# 開發指南

所有命令均在 repository 根目錄執行。產品操作見 [README](../README.md)，
代理開發政策見 [AGENTS.md](../AGENTS.md)。

## 環境與常用命令

首次設置：

```bash
uv sync --locked --all-extras
./scripts/install-git-hooks.sh
```

| 命令 | 用途 |
| --- | --- |
| `./scripts/rebuild-env.sh` | 重建 `.venv`，依 `uv.lock` 安裝鎖定版本。 |
| `./scripts/format.sh` | 自動修正 lint 與格式，會修改檔案。 |
| `./scripts/check-fast.sh` | 唯讀檢查 Ruff、格式、strict Mypy 與 Markdown。 |
| `uv run --all-extras pytest` | 完整測試，包含真實 Codex／MCP 策略生成。 |

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
審查與完整 pytest，並清理臨時產物。

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

- **環境**：腳本使用 Bash，目前未保證原生 Windows 相容；Git 需支援
  `merge-tree --write-tree`。更新依賴時執行 `uv lock`，並提交 lockfile 與設定變更。
- **登入與連線**：完整測試及文件審查需要已登入的 Codex CLI 與網路；可用
  `TRADINGDEV_CODEX_BIN` 指定 CLI。PR 檢查與清理另需 `gh` 能存取目標 repository。
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
