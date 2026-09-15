# AGENTS.md

## 政策與溝通

- 本檔是 repository 的唯一代理開發政策來源。其他代理入口只引導完整閱讀
  本檔，不另存一份政策。可執行規則以專案 scripts 與設定檔為準。
- 對話與完成回報使用繁體中文；程式碼、命令、識別字與 commit message
  可使用英文。
- 修改前說明目的與主要方向；工作中回報具體發現與設計取捨。
- 不得為了承載回覆新增計劃、測試說明或變更紀錄 Markdown 文件。既有
  README、架構與契約文件須隨其描述的行為更新。

## 設計原則

- 除使用者明確指定限制外，不以最小修改或向後相容限制設計；在授權任務
  範圍內，以架構、可讀性、可測試性與程式碼品質優先。
- 綜合考慮 SOLID、KISS、YAGNI、內聚性與低耦合，必要重構可納入任務。
- MCP tools、CLI、Python API、策略狀態與設定格式均可調整。不得把目前的
  名稱、模組配置或介面當成不可變條件；變更時同步更新使用端、測試與文件。
- 移除任務涉及的舊路徑時，不為假設的使用者保留 compatibility shim。
  不擴大清理與任務無關的程式碼。
- 實質擴大功能範圍或改動既有使用者資料前，先說明具體影響並確認授權。

## 產品邊界與架構

- 核心產品是讓 LLM 透過標準 MCP 協助使用者開發、驗證及研究策略。
  目前提供歷史回測、參數最佳化與結果追蹤，不提供真實下單。
- 後端保持 MCP 通用性；Codex、Claude 或其他客戶端差異留在各自整合層。
- MCP、CLI、dashboard 共用 application services；交易與資料邏輯留在
  domain，檔案、資料庫、外部資料來源與程序操作由 adapters 負責。
- 當前架構見 `ARCHITECTURE.md`，策略契約見 `docs/strategy_contract.md`，
  執行產物見 `docs/run_artifacts.md`。不要在本檔複製易變的介面清單。
- 策略訊號須符合契約、不得修改輸入資料或使用未來資料。YAML 參數在
  驗證、dry-run 與實際執行間必須一致。
- Git 管理的 bundled strategies 與使用者 workspace 內的 generated
  strategies 分開。LLM 經 MCP 建立與修正生成策略是正常產品功能。
- 驗證中執行生成 Python 不等於安全隔離；workspace 參數也不是 sandbox。
  不得將目前的靜態檢查或執行限制描述為完整 sandbox。

## 環境與品質工具

- 使用 repository 的虛擬環境；開發環境由 `uv sync --locked --all-extras` 建立。
  Python 命令使用 `uv run python`，共用 scripts 使用同一個 `.venv` interpreter。
- `uv.lock` 納入 Git；跨平台套件差異以依賴條件與實際平台測試處理。
  `scripts/rebuild-env.sh` 先檢查 lock，再重建本 repository 的 `.venv` 並安裝
  鎖定版本，不刪除 lock 或清空共用 uv 快取。更新依賴須明確更新 lock、檢查
  差異並測試，不以重建環境隱式升級套件。
- Ruff lint、Ruff formatter 與 strict Mypy 的唯一規則來源是 `pyproject.toml`。
  IDE 與 CLI 同步；生成策略使用同一份政策，wheel 收錄該設定。
- 不使用 Black、獨立 `mypy.ini` 或另一份較寬鬆的生成策略規則。
  型別依賴例外須逐模組指定並有依據，不得全域放寬以讓檢查通過。
- `scripts/format.sh` 明確執行會修改檔案的 fixer／formatter。
- `scripts/check-fast.sh` 執行唯讀 Ruff、格式、strict Mypy 與 Markdown 檢查。
  Markdown 使用專案 PyMarkdown；中文敘述與表格不套用 ASCII 行寬限制。
- 不依賴系統全域安裝的品質工具，不使用代理專屬 Stop hooks 重複執行檢查。
- Python 版本、依賴與 project version 以 `pyproject.toml` 為準；不要直接移植
  其他 repository 的版本或 dependency audit 政策。

## 測試與清理

- `pytest`／`uv run pytest` 預設執行完整套件，包含真實 Codex 經 MCP 撰寫
  策略的測試。需要已登入 Codex CLI 及網路；缺少條件須失敗，不得默默略過。
- 行為變更更新相關測試；bug fix 加入能重現問題的 regression test。驗證
  正常、邊界與失敗路徑，避免只重述實作的測試。
- 清楚區分真實 LLM、真實 MCP／worker、service 與替代外部服務的測試。
  不得把所有 pytest cases 都稱為整合測試。
- 新增或修改的測試若產生策略、行情、資料庫、結果、日誌或工具快取，
  使用會立即刪除的獨立臨時目錄；不可寫入一般使用者 workspace。
- 成功、例外與逾時都須清理檔案及測試啟動的程序。先確認程序身分並停止，
  再刪除目錄。清理失敗須回報，且須有正常／失敗／逾時的清理回歸測試。
- 固定隨機種子與合理數值誤差；不以重跑掩蓋 flaky failure。
- 不為通過檢查而增加 skip／xfail；必要例外須明確說明理由與未驗證範圍。

## 工作樹與分階段提交

- 唯讀分析不建立 branch；修改前用 `scripts/detect-primary-branch.sh` 判定
  主線，建立專用 task branch。主線只接受 merge，不可直接 commit 或 rebase。
- 不得 stash、reset、clean、覆寫或混入既有使用者修改。若工作樹包含與任務
  無關的修改，使用獨立 worktree，不擅自搬移那些修改。
- 開始實作時辨識可獨立檢查的開發階段；每完成一個有意義的階段，執行
  相關檢查並建立 commit。不要等整個任務結束才一次提交全部變更。
- commit 按問題與責任切分，不按任意檔案數切碎相互依賴的修改；小而內聚
  的任務可以只有一個 commit。不得為了分段留下已知失敗的階段。
- 使用 `scripts/git-flow-commit.sh "type: message" [files...]` 提交階段。
  此腳本只提交目前 task branch；不會立即合併或刪除分支。
- 非 merge commit 使用 Conventional Commits；不相容變更標註 `!` 或
  `BREAKING CHANGE:`，並在對話交代具體影響。
- Merge commit 只能建立於主線；task branch 若需同步主線，可 rebase 該
  task branch。整合只要求共同祖先，不要求先包含主線最新提交。
- 任務完成且工作樹乾淨後，使用 `scripts/git-flow-merge.sh` 整合。它以
  `--no-ff` 合併至主線；失敗會 abort 並保留 task branch，成功才移除專用
  worktree（若有）及以 `branch -d` 刪除已合併分支。
- 授權的開發流程包含上述本機分階段 commit 與整合；push、遠端分支、tag、
  publish、deploy 與 force 操作仍需使用者明確授權。不得用 `--no-verify`。

## Git hooks 與自動文件審查

- 用 `scripts/install-git-hooks.sh` 安裝 repository 的 `.githooks`。
- 每次 commit 只跑快速檢查與提交格式檢查。快速檢查驗證暫存內容的獨立
  snapshot，不能用尚未暫存的修正掩蓋待提交版本的錯誤。
- 合併主線前，自動由 Codex 比對候選程式差異與既有架構／契約／操作文件，
  然後執行 `scripts/check-full.sh`（快速檢查加完整 pytest，包含真實 Codex）。
- 文件審查使用唯讀、無工具的 Codex 執行，僅傳入 Git 版本的程式與文件，
  不修改 repository。發現過時文件、執行失敗、逾時或無效回覆皆阻止合併。
  修正文件後正常提交，再重新合併；不在 hook 中偷偷修改或提交文件。
- 證據超過 reviewer 的明確容量限制時直接失敗，不截斷後假裝完整審查。
  LLM 語意審查可能漏判，不能當成文件正確性的形式證明。
- 完整檢查驗證合併後的確切 Git tree。成功紀錄放在 Git metadata；相同
  內容與檢查環境可重用，主線 push 也會查核紀錄。不得修改或偽造紀錄。
- `scripts/git_gate.py full` 可手動驗證已提交且乾淨的 task branch。一次
  snapshot 檢查執行逾時為 900 秒，文件審查為 180 秒，程序清理另計；
  這不是整個開發任務時限。
- 測試、審查或候選內容在檢查中改變時不得視為通過。Hook 不執行遠端發布。

## 完成回報

- 說明改了什麼、原因與設計取捨，包括功能和結構的變化。
- 說明是否受最小修改／向後相容限制，及實際相容性影響；不能只說「重構」。
- 列出執行的檢查、結果及未涵蓋範圍。對清理的主張必須有對應證據。
- 列出各階段 commit、主線 merge 與 branch／worktree 清理狀態，說明是否
  已 push。不要只回覆「已完成」或只提供一個通過的測試總數。
