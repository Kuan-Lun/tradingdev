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
3. `save_strategy`：建立新 draft revision，保留回傳的 `revision_id`。
4. `validate_strategy`：跑 syntax、static policy、ruff、mypy、繼承與 signal
   contract 檢查；傳入剛保存的 `revision_id`。
5. `dry_run_strategy`：傳入相同 `revision_id`，只接受 validated revision，
   通過後升為 runnable。
6. `start_backtest` 或 `start_walk_forward`：傳入相同 `revision_id`，
   只接受 runnable/promoted revision。
7. `get_job_status`、`list_runs`、`get_run` 讀取摘要，再用
   `get_metric_catalog` 探索指標、`get_run_metrics` 查詢完整結果；
   `compare_runs` 比較已保存結果，`list_artifacts` 取得原始紀錄。

`get_strategy`、驗證、dry-run、promote 與三種執行工具都接受 `revision_id`。
省略時，每次操作只選取一次當前 revision；已建立的工作固定使用選定版本。
修正策略須重新 `save_strategy`，新 revision 從 draft 開始；舊版本與驗證證據保留，
可用其 `revision_id` 查詢或執行。job、run 與策略回覆會帶回所選版本。
參數實驗沿用已 runnable 的 revision：在 `start_backtest` 或 `start_walk_forward`
傳入 `parameters`，只覆寫該次執行的策略參數，巢狀物件逐層合併，其餘保留基礎值。
覆寫只能指定基礎參數或建構子預設值中已有的鍵；巢狀映射的未知鍵同樣會被拒絕，
回傳 `invalid_execution_request`，不建立工作。
每次執行固定完整設定，並以有效參數執行短、長兩組訊號契約檢查；
不修改原 Python、基礎 YAML、驗證證據或 current pointer，也不建立新的 revision。
例如同一 MACD revision 可分別傳入 `parameters={"fast_period": 12, "slow_period": 29}`
及其他組合，結果各自保存成 run。修改程式或要保存新的基礎設定時才另存 revision。
最佳化可覆寫搜尋範圍內的參數，其餘保留基礎值。
Bundled 策略仍由 Git 管理，`revision_id` 為 `null`。

提交生成策略時，即使沒有覆寫參數，也會先在 MCP server 程序內建構策略並執行
訊號契約檢查；這發生於建立 job、啟動 worker 之前，不受 worker 監督或試跑逾時管控。
詳見[策略安全模型](docs/strategy_contract.md#security-model)。

`cleanup_strategy_drafts(strategy_id)` 預設只預覽舊草稿，回傳每個 revision
是否可清理及保留原因。先向使用者說明清單；取得明確刪除授權後，才以
`cleanup_strategy_drafts(strategy_id, revision_ids=[...], apply=true)` 清理指定項目。
不存在的生成策略會回傳 `strategy_not_found`，不會為該次請求建立鎖檔。
只接受非 current、未被歷史工作或執行引用的 draft；validated、runnable、promoted
均保留。套用時會重新檢查，不能把先前預覽視為永久有效的刪除資格。
歷史資料不明或損壞時會阻止清理，檔案完整性有疑慮的版本也會保留。
工具不會自動清理、合併既有版本，或改寫歷史回測；刪除遇到檔案系統錯誤時逐項回報，
可能已刪除部分檔案，須依回覆確認結果。

每次提交回測、walk-forward 或最佳化時，會先固定完整執行規格
`manifest.json`，包含策略版本、有效設定與預設值、資料路徑，以及最佳化的搜尋
與確認設定。成功回覆的 `manifest_hash` 可與 job、run 及執行規格產物核對；
worker 使用這份規格，稍後修改原 YAML 或切換當前策略版本都不會改變已提交工作。
`config.yaml` 是方便檢視的設定副本；修改它不會修改工作。要改設定請提交新工作。
策略宣告的身分與非參數欄位保留供 revision 核對；建構子參數與內建策略模型的預設值另外展開並
存進 `strategy_execution`。執行時不再補入新的預設值；不相容的模型或參數會令
工作失敗。規格使用 schema 4，固定完整設定結構、頂層執行種子、績效計算設定及最佳化方向；
早期 schema 1／2／3 規格須重新提交，不會原地遷移。格式詳見
[執行規格契約](docs/run_artifacts.md#execution-manifest)。
規格固定執行請求，但沒有封存行情內容、Python 依賴或隨機數產生器狀態，
因此不保證日後能得到完全相同的結果。

舊版 `generated_strategies/<id>.py`／`<id>.json` 與 `configs/<id>.yaml`
不會自動遷移或改寫。`list_strategies` 會列為 `kind: legacy`、
`status: revision_required`，不影響其他策略的探索；以 `get_strategy` 讀取原程式與
YAML，再明確透過 `save_strategy` 保存並完成 validate 與 dry-run。
舊的 runnable/promoted 狀態不能替新 revision 授權；保存後清單改列新版本，
舊檔仍保留。若原程式或 YAML 缺失、無法讀取，請還原檔案或提供替代內容再保存。
若舊策略與 bundled 策略同名，預設查詢與執行選擇 bundled；請用
`get_strategy(strategy_id, legacy=true)` 讀取舊內容，改用另一個未保留的 ID 保存。

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
| Strategy | `get_strategy_contract`, `list_strategies`, `get_strategy`, `save_strategy`, `validate_strategy`, `dry_run_strategy`, `promote_strategy`, `cleanup_strategy_drafts` |
| Data | `list_data_sources`, `list_available_data`, `inspect_dataset`, `ensure_data` |
| Backtest | `start_backtest`, `start_walk_forward` |
| Optimization | `start_optimization`, `confirm_optimization` |
| Jobs/Runs | `get_job_status`, `list_jobs`, `cancel_job`, `list_runs`, `get_run`, `get_metric_catalog`, `get_run_metrics`, `compare_runs` |
| Artifacts | `list_artifacts`, `get_artifact` |
| Requests | `record_feature_request`, `list_feature_requests` |

每個工具都提供具體的 `outputSchema` 與唯讀、破壞性、冪等及外部互動提示。
輸入 schema 的頂層參數不接受未知名稱；例如將 `revision_id` 拼成
`revisions_id` 會在執行前回傳錯誤，必須更正後重新呼叫。
合法的選填參數仍可省略，策略參數等動態映射仍依各工具的 schema 傳入。
客戶端應依 `tools/list` 的 schema 讀取 `structuredContent`：單一物件回覆直接
位於其根層；清單與成功／失敗聯集回覆放在 `structuredContent.result`。
從舊版升級時，直接讀取根層 `success`、`job_id` 等欄位的自訂客戶端須相應調整；
工作區及既有策略、回測資料不會遷移或改寫。

預期的操作失敗保留 `success: false` 或空 `job_id` 等工具原有語意，並提供
穩定的 `code`；驗證未通過則查看 `diagnostics` 的 `code`、`phase` 與 `fix`。
`get_job_status` 的 `status: not_found` 表示查無工作；`status: failed` 表示
找到已失敗的工作。參數格式錯誤、未預期的執行例外或回覆不符合 schema 時，
MCP 回傳 `isError: true`。不應只依 MCP `isError` 判斷應用操作是否成功。

工具提示描述實際副作用，不是權限檢查或安全隔離。`get_job_status` 可能更新
失去 worker 的工作狀態；`ensure_data` 可能下載資料並替換不完整快取；
策略驗證與 dry-run 會執行生成 Python。
`destructiveHint: false` 代表只做新增；會替換既有狀態或驗證證據的工具也會
標示為 `true`，不只限於刪除檔案。

最佳化先以 `start_optimization` 試跑並估時，等使用者同意後才呼叫
`confirm_optimization` 執行完整搜尋；以 `get_job_status`／`get_run` 查詢結果。
進入 `pending_confirmation` 後最多等待 30 分鐘；逾時工作會標示為 `failed`，
不會執行剩餘搜尋。
訓練與測試日期都包含終日，須符合 `train_start < train_end < test_start < test_end`。
呼叫指定的交易對與時間框架會覆寫 YAML；搜尋參數以外的 YAML 固定參數仍會保留。
搜尋依指標定義選擇最大值或最小值，例如回撤越小越好；指標必須適用於執行模式。
缺少年化設定不能以年化指標最佳化；粗於日頻或無法辨識的 K 棒頻率，不能選擇
需要日級觀測的指標。這些請求會在建立工作前被拒絕。
不可計算的候選值不參與排名，全不可用時工作失敗。
參數名稱排序後展開組合，各參數的候選值順序保留。
巢狀參數只覆寫搜尋值指定的欄位，其他欄位保留已固定的值。提交前會檢查候選組合，
不符合參數結構或內建設定模型契約的組合不會建立工作；策略建構與執行錯誤仍可能
於試跑時發生。
含 `validation:` 設定的策略 config 須使用 walk-forward；最佳化使用自己的訓練／
測試日期，會以 `invalid_optimization_request` 拒絕同時提供兩套切分設定。
升級前沒有執行規格的歷史結果仍可查詢，但舊工作不能以新版 worker 接續執行或確認；
請重新提交工作。

## 績效指標與計算設定

報酬與風險指標由 `empyrical-reloaded` 計算，逐筆交易統計由 `vectorbt` 計算。
日／月損益、費用與成交量來自同一份執行帳本。完整指標會保存，不因摘要欄位選擇而刪除。

整次執行的 `random_seed` 只放在 YAML 頂層（與 `strategy`、`backtest` 同層），
接受 0 至 4294967295 的整數或 `null`；舊的 `backtest.random_seed` 與未知設定名稱
會被拒絕。策略可使用 `tradingdev.domain.randomness` 的獨立亂數產生器，
或以 `get_seed()` 明確設定第三方模型。它不會覆寫 Python／NumPy 全域亂數狀態，
也不會自動控制任意第三方套件的亂數。
`get_strategy_contract` 提供完整設定 schema；驗證與 dry-run 回覆的 `effective_config`
可用來核對種子與其他設定是否符合要求，再啟動回測。

在 YAML 的 `backtest` 中明確設定日報酬的年化頻率：

```yaml
random_seed: 42
backtest:
  # 其餘 symbol、timeframe、日期、資金等設定依策略契約填寫
  periods_per_year: 365.0 # 全年交易的加密貨幣；股票等市場需選擇合適交易日數
  risk_free_rate: 0.0 # 年化小數利率
  required_return: 0.0 # Sortino 使用的年化目標報酬
```

年化統計使用觀察到的 UTC 日報酬，與策略的 K 棒頻率分開；缺資料日不自動補零。
未指定 `periods_per_year` 時，年化報酬、Sharpe、Sortino 等指標為不可用，並非零。
日級觀測需要日線或更細的 K 棒。週線、月線、多日 K 棒或無法辨識的頻率，
不會被當成日報酬，也不會插值補出日淨值；年化指標、`daily_max_drawdown` 與
日／月損益統計會回傳 `null` 及原因，CLI 顯示 `N/A`。月損益同樣需要日級觀測，
避免將跨月 K 棒的整段損益歸入單一月份。`periods_per_year` 仍指一年中的日數，
不能改填 52 來套用週線。總報酬、總損益、K 棒回撤與交易統計仍可計算。
最大回撤保留逐根 K 棒解析度；`max_drawdown` 是非負比例，
`max_drawdown_amount` 是非負金額。Calmar 的分母是日報酬曲線回撤，
可由 `daily_max_drawdown` 核對。Volume 模式沒有初始資金，應使用金額損益與回撤，
資金報酬率指標不適用。`total_trades`、`win_rate`、`profit_factor` 與
`trade_expectancy` 只統計已平倉交易，包含進出場成本；未平倉交易另列。

新口徑會改變既有數字與交易數，升級後須重新回測才有新結果；既有使用者結果不會自動改寫。

對話中的 run／job 回覆只提供摘要，並列出可查詢的指標與 `available_scopes`。
例如在完成回測後，LLM 可呼叫：

```text
get_metric_catalog(mode="signal")
get_run_metrics(run_id="<run_id>", metric_ids=["daily_pnl_mean", "total_volume"])
get_run_metrics(run_id="<run_id>", scope="fold/0/test")
```

第二行讀取預設範圍的指定指標；第三行讀取 walk-forward 第 0 個 fold 的完整測試指標。
實際 scope 以該 run 回傳清單為準。單次回測的預設範圍是 `full`，walk-forward 是
`test_summary`，最佳化是選定參數的樣本外 `test`。Fold 摘要是各 fold 的統計分布，
包含有效樣本數，不能當成整段期間重新計算的績效。

詳細結果附保存時的定義、單位、套件版本、設定，以及 `null` 的原因。
`compare_runs` 會指出口徑或設定不同造成的可比性限制，不自動排名。
`performance.json` 保存完整指標，`observations.json` 保存淨值、報酬、時間與交易紀錄；
可透過 `list_artifacts`／`get_artifact` 讀取。歷史 run 若缺少新 artifact，仍可讀原結果，
詳細查詢則明確回報不可用，不會猜測來源或重新計算。完整格式見
[執行產物契約](docs/run_artifacts.md)。

## 工作區與檔案

- 內建策略：
  `src/tradingdev/domain/strategies/bundled/<strategy>/strategy.py`
- 內建策略設定：
  `src/tradingdev/domain/strategies/bundled/<strategy>/config.yaml`
- 透過 MCP 產生的策略：
  `workspace/generated_strategies/<strategy_id>/revisions/<revision_id>/strategy.py`
- 生成策略的設定：
  同一 revision 目錄的 `config.yaml`；`metadata.json` 保存狀態與驗證證據。
- 生成策略的當前版本指標：
  `workspace/generated_strategies/<strategy_id>/current.json`
- 行情資料快取：
  `workspace/data/raw/` 與 `workspace/data/processed/`
- 工作狀態與執行結果索引：
  `workspace/tradingdev.sqlite`
- 背景工作的規格與結果檔案：
  `workspace/runs/<run_id>/`，包含固定執行規格 `manifest.json`、結果、
  使用的設定與策略副本、資料集識別資訊及 dashboard 所需檔案。

`workspace/` 是預設工作區；指定其他 `--workspace` 路徑時，生成策略、
資料與執行結果會存到該路徑。內建策略與設定隨專案提供，透過 MCP 撰寫的
策略另存於工作區。

## CLI

也可以在終端機直接執行策略設定檔：

```bash
uv run python -m tradingdev --config \
  src/tradingdev/domain/strategies/bundled/kd_strategy/config.yaml
```

Generated 策略請指定已通過 dry-run 的 revision 所屬 `config.yaml`，
或複製它建立執行設定，調整 `strategy.parameters` 進行參數實驗。
`strategy` 的身分、`description`、`version`、`fit` 等其餘原有欄位須完整保留；
新增、移除或變更這些欄位需要另存並驗證新 revision。
`source_path` 必須仍指向原 revision 的來源；`source_hash` 由執行流程驗證後填入。
交易對、期間與成本可在 `strategy` 以外的設定區段調整。
CLI 也先固定執行規格，再執行並用該規格保存結果快取；執行中修改原設定檔，
仍可保存原規格的結果。每次執行保存成獨立 run，即使設定相同也不覆寫舊結果。
CLI 的 `manifest.json` 存於 `workspace/runs/cli_<cache_key>_<execution_id>/`，
pipeline 結果快取則位於資料目錄的 `processed/cache/<run_id>.pkl`。
`cache_key` 記錄規格、資料與程式指紋；清除快取後重跑會建立新的完整結果。

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

歷史 run 的指標可透過 `get_run` 查詢，不代表新版 Dashboard 能完整顯示。
Dashboard 需要可載入的 `pipeline_result` 產物，且設定快照須符合現行格式；
含 `backtest.random_seed` 等已移除欄位的舊快照會載入失敗。舊交易若缺少
`status`、進出場時間（或可對應行情時間的 `entry_idx`／`exit_idx`），或
`exit_notional`，交易圖表與月成交量可能缺漏。需要完整 Dashboard 結果時，
請以新版設定重新回測；既有產物不會自動遷移。

## 相關文件

- [策略撰寫契約](docs/strategy_contract.md)
- [執行結果與產物](docs/run_artifacts.md)
- [內建策略說明](docs/strategies/)
- [開發指南](docs/development.md)：開發環境重建、品質檢查、測試、Git／PR 流程與分支清理。
- [專案架構](ARCHITECTURE.md)
- [代理開發政策](AGENTS.md)
