# 開發指南

本文件供修改及維護 TradingDev 專案程式碼的開發者使用。安裝 MCP、使用工具
開發交易策略及查看結果，請見 [使用指南](../README.md)。所有命令均在本機
repository 根目錄執行。

## 建立與重建開發環境

安裝鎖定的 runtime、開發工具及 dashboard 依賴：

```bash
uv sync --locked --all-extras
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

## 品質檢查與測試

```bash
./scripts/format.sh
./scripts/check-fast.sh
uv run --all-extras pytest
```

`format.sh` 會修改格式及套用可自動修正的 lint 項目；`check-fast.sh` 唯讀
檢查 Ruff、格式、strict Mypy 與 Markdown。Ruff、formatter 與 Mypy 的規則
統一放在 `pyproject.toml`。

`pytest` 預設執行全部單元、MCP 整合及真實 Codex 端對端測試。
需有已登入的 Codex CLI 與模型網路連線；測試會自動啟動 MCP，並清除臨時產物。
CLI 會從 `PATH` 或 macOS 的 VS Code 擴充套件尋找，也可用
`TRADINGDEV_CODEX_BIN` 指定。缺少 CLI、登入或連線時，測試會回報失敗。

## Git hooks 與分階段提交

先安裝本 repository 的共用 Git hooks：

```bash
./scripts/install-git-hooks.sh
```

開發政策見 [AGENTS.md](../AGENTS.md)。在 task branch 完成一個可驗證階段後，
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

## 審閱者在本機檢查 PR

負責決定合併的人，在已提交且乾淨的 checkout 明確執行下列命令。
需要 Git 支援 `merge-tree --write-tree`、本機 `gh` 已登入且能讀取 repository，
以及已登入的 Codex CLI、連線及本專案完整開發環境；不需要雲端 Codex CI 或把憑證交給別人。

PR 檢查與清理命令目前只支援 `github.com`，`gh` 必須登入該 host。
Remote 可使用 `https://github.com/OWNER/REPO.git`、
`git@github.com:OWNER/REPO.git` 或 `ssh://git@github.com/OWNER/REPO.git`；
僅接受預設連接埠（HTTPS 443／SSH 22），不支援 Enterprise host、SSH alias、
本機路徑或含登入憑證的 HTTPS URL。

待檢查的 PR 必須仍開啟、尚未合併、來源 repository 可辨識，且目標是本機
偵測出的主線。`--remote` 只指定 PR 目標 repository，不會更改前述主線
偵測規則；需要其他主線名稱時設定 `git config tradingdev.primaryBranch <branch>`。

### 取得 PR 編號並執行檢查

PR 編號是 GitHub 在建立 Pull Request 時分配的整數，顯示在 PR 標題旁的
`#編號`，也出現在網址結尾的 `/pull/編號`。例如，網址結尾為 `/pull/27`
的 PR，其編號就是 `27`。

操作順序如下：

1. 開發者在開發分支完成階段性 commit，將分支 push 到 GitHub。
2. 開發者在 GitHub 的 Pull requests 頁面，從該分支向目標主線建立 PR。
3. 審閱者打開這個 PR，從標題或網址取得編號。
4. 審閱者在本機 repository 根目錄執行 `check-pr.sh`，傳入該編號。
5. 檢查通過後，審閱者在 GitHub 留下評論、核對版本並決定是否合併。

若 PR 已經存在，直接從第 3 步開始。也可以在本機 repository 根目錄列出
目前開啟的 PR，從輸出的 `number` 欄位取得編號：

```bash
gh pr list --state open --json number,title,url
```

以下範例假設要檢查的是 PR #27；請將 `27` 換成你實際取得的 PR 編號：

```bash
./scripts/check-pr.sh 27
```

數字 `27` 僅為範例，不代表本專案目前存在這個 PR；傳入的是 PR 編號，
不是分支名稱或 commit SHA。腳本預設查詢 `origin` 對應的 repository。
若 PR 的目標是 `upstream`，可執行 `./scripts/check-pr.sh 27 --remote upstream`；
使用 `gh pr list` 查詢時，也要加上 `--repo OWNER/REPO` 指定同一個目標
repository，其中 `OWNER/REPO` 是該 remote 在 GitHub 上的擁有者與專案名稱。
只有本機 commit、尚未建立 PR 時，請使用本節後面的本機候選檢查命令。

腳本讀取指定 PR，在獨立臨時 repository fetch 當前主線與 PR head，計算
合併候選內容。版本不一致或合併衝突會立即失敗；它不切換目前分支。
接著執行 Codex 程式／文件一致性審查及 `scripts/check-full.sh`，包含快速
檢查和完整 pytest（真實 Codex 經 MCP 撰寫策略的測試也在內）。Ruff／Mypy
需要 dashboard extra，以免缺少套件型別掩蓋問題。

文件審查只讀 Git 內容，不修改文件；過時文件、錯誤或逾時會令命令失敗。
每次執行都重新檢查，不使用舊版完整檢查快取。臨時 repository、snapshot
與測試產物在結束時清理；快照執行上限 900 秒，文件審查上限 180 秒，
程序清理時間另計。Codex 語意審查仍可能漏判。

結束時再次核對 PR 身分／狀態與實際遠端 refs；PR API 的 base SHA 可能
落後主線，因此不作為即時版本依據。輸出 URL、base／head SHA、候選 tree
與結果，供審閱者自行貼到 PR。腳本不發送評論或合併。網頁合併前必須核對目前 base／head；
任一改變便重新檢查。可在自己的終端機讀取目前版本：

```bash
# 延續 PR #27 的範例；請替換為實際主線名稱、PR 編號及 remote。
git ls-remote origin refs/heads/main refs/pull/27/head
```

純本機檢查無法知道使用者何時按下 GitHub 合併，也不能阻止核對後的版本
變更；目前靠審閱者遵守此流程，沒有遠端強制檢查門禁。

尚未建立 PR 時，可檢查本機指定主線與目前提交的合併候選：

```bash
uv run --no-sync python scripts/git_gate.py full --base main
```

這不驗證遠端 PR；`--base` 必須明確指定，也可加 `--head <ref>`。

## PR 合併後清理本機分支

在 GitHub 合併後，切換到其他分支，再明確指定已完成的 PR 與本機分支。
以下延續 PR #27 的範例，假設要清理的本機分支叫 `feature/my-task`：

```bash
./scripts/cleanup-pr.sh 27 --branch feature/my-task
```

請將 `27` 換成已合併的 PR 編號，將 `feature/my-task` 換成要刪除的實際
本機分支名稱；這兩個參數分別指定 PR 與本機分支，不能互換。

可加 `--remote upstream` 指定 PR 目標 repository。腳本只 fetch 目標主線，
不執行 pull。PR 也必須以本機偵測出的主線為目標。它核對 PR 已合併、
本機 upstream repository／分支與 PR 來源相符、本機 tip 等於 PR head，以及 PR head／merge commit 都已在
取得的主線歷史中，才原子刪除指定的本機分支 ref。檢查後若分支增加提交，
刪除會失敗。它不依靠 commit message 的分支名稱，也不呼叫 LLM。

主線與 `git config --add tradingdev.protectedBranch <branch>` 指定的分支
會保留；正在任何 worktree checkout 的分支也不刪除。若 GitHub 使用
squash／rebase 而無法證明原始 head 已包含於主線，腳本保留分支並要求人工
確認；fetch 不會恢復改寫過的祖先關係。

遠端分支、worktree 與本機 branch config 都保留。保留 config 是為了避免
ref 刪除後與同名新分支競態而誤刪設定；命令結果會明確列出此行為。
