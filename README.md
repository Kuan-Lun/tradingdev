# TradingDev

用對話把交易想法變成可驗證的策略。TradingDev 讓你的 AI 助手撰寫策略、
檢查程式、執行歷史回測、比較參數，並保存每次研究的設定與結果。
目前提供歷史研究功能，不提供真實下單。

你可以從內建策略開始，也可以直接描述自己的規則。助手透過 MCP
（讓 AI 呼叫外部工具的標準協定）操作 TradingDev；回測在你的電腦上執行。
TradingDev 本身不提供對話模型，需搭配支援 MCP 工具呼叫的 AI 客戶端。

- [安裝與連接助手](#安裝與連接助手)
- [完成第一次回測](#完成第一次回測)
- [建立與研究自己的策略](#建立與研究自己的策略)
- [查看結果與圖表](#查看結果與圖表)
- [資料與工作區](#資料與工作區)
- [終端機與 HTTP 用法](#終端機與-http-用法)
- [常見問題](#常見問題)
- [相關文件](#相關文件)

## 安裝與連接助手

### 1. 準備環境並安裝

你需要 macOS 或 Linux、Git、[uv](https://docs.astral.sh/uv/getting-started/installation/)，
以及支援本機 MCP（stdio）的 AI 客戶端。專案使用 Python 3.12 或 3.13；
原生 Windows 目前不支援背景回測與最佳化。

首次下載與安裝：

```bash
git clone https://github.com/Kuan-Lun/tradingdev.git
cd tradingdev
uv sync --locked
```

已有專案時，直接在現有專案根目錄執行 `uv sync --locked` 即可。
這會建立 `.venv` 並安裝鎖定版本的套件，包含內建策略需要的機器學習依賴；
首次安裝可能需要較長時間。若要指定 Python 版本，使用
`uv sync --locked --python 3.13`。

以下終端機命令皆在專案根目錄執行。

### 2. 將 TradingDev 加入 AI 客戶端

在客戶端的 MCP 設定中新增一個本機 stdio server，名稱填 `tradingdev`，
啟動命令填 `uv`。支援 `mcpServers` JSON 格式的客戶端可使用：

```json
{
  "mcpServers": {
    "tradingdev": {
      "command": "uv",
      "args": [
        "--directory",
        "/absolute/path/to/tradingdev",
        "run",
        "--locked",
        "tradingdev-mcp",
        "--workspace",
        "/absolute/path/to/tradingdev-workspace"
      ]
    }
  }
}
```

請替換兩個絕對路徑：

- `/absolute/path/to/tradingdev`：剛下載的專案目錄；現有 checkout 名稱不同也可以。
- `/absolute/path/to/tradingdev-workspace`：你選擇存放策略、行情與回測結果的資料夾，
  TradingDev 會建立所需目錄。之後查詢結果與開啟圖表都使用同一工作區。

若客戶端以表單設定，將 `command` 與 `args` 分別填入命令與參數欄位；
設定檔位置與格式依客戶端而定。若它找不到 `uv`，在終端機執行
`command -v uv`，將輸出的完整路徑填入 `command`。

儲存後重新載入 MCP 設定或重啟客戶端，並啟用 TradingDev 工具。
客戶端會自行啟動 server，不需要另外開終端機常駐執行。

### 3. 確認助手能使用工具

在對話中輸入：

> 請使用 TradingDev 列出可用策略與行情資料來源，並說明哪個內建策略適合用來確認回測流程。

助手應實際呼叫工具，取得策略清單與資料來源。
清單會區分隨專案提供的內建策略，以及保存在工作區的自建策略。
若助手只提供一般建議，請先確認客戶端已連接 `tradingdev` 並允許工具呼叫。

## 完成第一次回測

先用內建 KD 交叉策略確認整個流程，不必先撰寫 Python：

> 請讀取 TradingDev 的內建策略 `kd_crossover`，說明它的交易規則、初始資金、
> 手續費與滑價設定。接著以 BTC/USDT、1 小時 K 線，回測 UTC
> 2024-01-01 00:00:00 到 2024-02-01 00:00:00。
> 請檢查所需資料，執行後追蹤工作直到完成，回報總損益、最大回撤、已平倉交易數，
> 並保留這次的 run ID 供後續比較。

第一次執行可能需要下載行情；之後可使用本機快取。
資料按年度取得，即使只回測一個月，也可能需要下載該年度的行情。
預設來源是 Binance Vision 的 USDT-M 期貨資料，範例中的 `BTC/USDT` 使用期貨行情。
資料是否可取得取決於來源、交易對與期間；下載失敗時可依[常見問題](#常見問題)檢查。

提交成功會取得 **job ID**，用來查進度或取消工作；完成並保存結果後，
以 **run ID** 查詢結果。背景回測目前使用相同的 ID，但提交成功不代表回測已完成。
助手可用 `get_job_status` 追蹤，完成後用 `get_run` 讀取結果。

接著可以追問：

> 請解讀剛才的結果，列出回測設定、成本與資料期間。哪些指標無法計算？
> 請說明原因，並找出值得進一步驗證的地方。

## 建立與研究自己的策略

### 把交易規則交給助手

描述交易標的、K 線頻率、進出場規則、期間、資金與成本即可開始。例如：

> 請建立名為 `btc_sma_cross` 的 BTC/USDT 均線策略：使用 1 小時 K 線，
> 10 期均線向上穿越 30 期均線時做多，向下穿越時平倉，不做空。
> 初始資金 10,000 USDT、單邊手續費 0.06%、滑價 0.05%，
> 回測期間為 UTC 2024-01-01 到 2024-07-01，年化日數設為 365。
> 請先讀取 TradingDev 策略契約，保存策略、完成驗證與試跑，再執行回測。
> 若檢查失敗，請依診斷修正，最後回報使用的策略版本、設定與結果。

助手會依序完成保存、驗證、試跑與回測。自建策略必須通過驗證及試跑才能執行；
這些檢查用來確認程式與訊號符合規則，不代表策略有獲利能力。

每次保存程式或新的基礎設定都會產生一個版本（`revision_id`）。
修正後的新版本需要重新檢查，舊版本與歷史結果仍保留；
可要求助手全程使用同一個 `revision_id`，避免混用版本。
自行撰寫 Python 或 YAML 時，請參考[策略契約](docs/strategy_contract.md)。

驗證、試跑與提交回測都可能執行生成的 Python。這些步驟不是完整的安全沙箱，
工作區路徑也不限制程式能存取的範圍；詳見[策略安全模型](docs/strategy_contract.md#security-model)。

### 比較參數

只改參數時，可以沿用已通過試跑的策略版本，不必複製程式：

> 請沿用 `kd_crossover`，分別以 `k_period` 為 9、14、21 執行三次回測。
> 交易對、期間、其他參數與成本都沿用第一次回測，完成後比較損益、回撤與交易數。
> 請列出各次 run ID，並說明結果是否適合直接比較。

每次執行會固定當次設定並保存獨立結果，不會改寫策略的基礎設定或先前的回測。
要修改已提交工作的設定，請另開一次回測。

### 使用樣本外驗證與最佳化

**Walk-forward** 將資料依時間分成多組訓練與測試區間，用後續未參與訓練的資料
觀察策略表現。請助手先設定切分方式，再執行 walk-forward；
含 `validation` 設定的策略必須使用此流程。

**參數最佳化** 搜尋指定的候選值，在訓練期間選出參數，再以獨立的測試期間評估。例如：

> 請對 `kd_crossover` 搜尋 `k_period` 為 9、14、21，`d_period` 為 3、5 的組合，
> 以 BTC/USDT、1 小時 K 線、總損益為目標。
> 訓練期間為 2024-01-01 至 2024-03-31，測試期間為 2024-04-01 至 2024-06-30。
> 請先試跑估時，告訴我組合數與預估耗時，等我同意後才進行完整搜尋。

最佳化會先試跑並等待確認；助手須在你同意後呼叫 `confirm_optimization`。
進入 `pending_confirmation` 後最多等待 30 分鐘，逾時會失敗，需要重新提交。
訓練與測試日期包含終日且不得重疊；最佳化使用自己的日期切分，不能同時套用
walk-forward 的 `validation` 設定。結果應同時查看訓練表現與樣本外表現。

## 查看結果與圖表

### 在對話中查詢

不需要記住所有工具名稱，可以直接請助手處理：

| 想做的事 | 可以這樣說 |
| --- | --- |
| 找回之前的結果 | 「列出最近的回測，顯示策略、期間與 run ID。」 |
| 追蹤或停止工作 | 「查詢這個 job ID 的進度。」或「取消這個 job ID 的工作。」 |
| 查摘要以外的指標 | 「查詢這個 run 的所有可用指標，說明定義、單位與無法計算的原因。」 |
| 比較兩次實驗 | 「比較這兩個 run 的損益與回撤，先檢查設定與計算口徑是否相同。」 |
| 檢查實際設定 | 「列出這個 run 保存的產物，讀取執行設定與策略版本。」 |

解讀結果時請留意：

- `N/A` 或 `null` 代表無法計算或不適用，不是零；可請助手查詢原因。
- 年化報酬、Sharpe 等指標需要明確設定 `backtest.periods_per_year`，
  例如全年交易市場可用 365。它表示每年日數，與 1 小時或 5 分鐘 K 線分開。
  週線、月線等粗於日頻的資料不提供年化及日／月損益統計。
- Volume 模式沒有初始資金基準，應查看金額損益、成交量與回撤金額。
  交易數、勝率等交易統計只計算已平倉交易。
- Walk-forward 摘要是各測試區間的統計分布，不能當成整段期間的單一績效；
  比較時可要求逐區間查看。最佳化則應確認查看的是訓練或樣本外測試結果。

執行時的設定與結果會保留，但行情、套件及環境未完整封存，
日後重跑不保證得到完全相同的數字。詳細定義見[執行結果與產物](docs/run_artifacts.md)。

### 開啟 Dashboard

Dashboard 可顯示已保存的一般回測與 walk-forward 結果。先安裝圖表依賴，
並將工作區設為 MCP 設定使用的同一路徑：

```bash
uv sync --locked --extra dashboard
export TRADINGDEV_WORKSPACE="/absolute/path/to/tradingdev-workspace"
uv run --locked --extra dashboard streamlit run src/tradingdev/adapters/dashboard/app.py
```

開啟終端機顯示的本機網址，從側邊欄選擇結果。也可以直接指定 run ID：

```bash
uv run --locked --extra dashboard streamlit run src/tradingdev/adapters/dashboard/app.py \
  -- --run-id <run_id>
```

請將 `<run_id>` 換成助手查到的實際值。最佳化結果請透過對話查詢，
目前不產生 Dashboard 所需的圖表產物。結束圖表服務時，在終端機按 `Ctrl+C`。

## 資料與工作區

行情會按策略設定從資料來源取得並快取。可請助手先列出來源，再檢查已有的資料；
`inspect_dataset` 只檢查本機資料，`ensure_data` 才會取得缺少的行情。
回測也可能自行下載資料，並替換不完整的快取。
需要波動率、資金費率等額外特徵的策略，還須準備設定中指定的資料。
同一交易對若要改用不同來源或現貨／期貨市場，請使用獨立的工作區與資料目錄，
避免沿用原來的行情快取。

工作區保存你的研究內容，與隨專案提供的內建策略分開：

| 位置（相對於工作區） | 內容 |
| --- | --- |
| `generated_strategies/` | 自建策略的各版本、設定與驗證紀錄 |
| `data/` | 行情與處理後資料快取，也包含 CLI 圖表結果快取 |
| `runs/` | 每次執行的設定、結果與相關產物 |
| `tradingdev.sqlite` | 工作、結果與產物的查詢索引 |

MCP 的 `--workspace` 優先於環境變數 `TRADINGDEV_WORKSPACE`；兩者都未設定時，
使用啟動目錄下的 `workspace/`。CLI 與 Dashboard 使用該環境變數，沒有
`--workspace` 參數。每次開新終端機時，請重新設定環境變數，或在自己的 shell
設定中保存它。

保留研究結果時，請保留工作區的資料庫與檔案，不要只留下 run ID 或單一結果檔。
若另外指定外部資料或快取目錄，也需保留那些路徑下的檔案。
要清理舊草稿，可請助手先預覽可刪版本，確認清單後再授權刪除；
清理工具會保留目前版本、通過驗證的版本及被歷史執行引用的版本。

## 終端機與 HTTP 用法

### 直接執行設定檔

不透過 AI 也可以執行回測。先設定與 MCP 相同的工作區，再使用內建 KD 範例：

```bash
export TRADINGDEV_WORKSPACE="/absolute/path/to/tradingdev-workspace"
uv run --locked python -m tradingdev --config \
  src/tradingdev/domain/strategies/bundled/kd_strategy/config.yaml
```

這會依檔案中的期間與成本執行，並在終端機顯示結果。每次執行都保存獨立 run，
之後可由助手查詢，或在 Dashboard 選取。
若要調整內建範例，先複製 YAML 作為自己的執行設定，再修改交易對、日期與參數。
含 `validation` 的設定必須加上 `--walk-forward`，例如：

```bash
uv run --locked python -m tradingdev --config \
  src/tradingdev/domain/strategies/bundled/xgboost_strategy/config.yaml \
  --walk-forward
```

自建策略須使用已通過試跑的版本。可複製該版本的 YAML 調整
`strategy.parameters`，但保留其餘策略宣告與指向原版本的 `source_path`；
交易對、日期與成本則在 `strategy` 區段以外調整。
修改策略程式或其他策略宣告時，請讓助手另存版本並重新驗證，
完整規則見[策略生命週期](docs/strategy_contract.md#lifecycle)。

### 使用 HTTP 連接

若客戶端使用本機 Streamable HTTP 連線，可手動啟動：

```bash
uv run --locked tradingdev-mcp \
  --workspace /absolute/path/to/tradingdev-workspace \
  --web --transport streamable-http --port 8000
```

將客戶端的 MCP URL 設為 `http://127.0.0.1:8000/mcp`，並保持此終端機執行。
這是本機工具端點，不是瀏覽器操作介面；圖表請使用 Dashboard。
只支援雲端 URL 的客戶端無法透過這個本機位址連接。

## 常見問題

| 情況 | 處理方式 |
| --- | --- |
| 客戶端找不到 `uv` 或 server 無法啟動 | 用 `command -v uv` 取得完整路徑；確認 `--directory` 指向專案根目錄，並已成功執行 `uv sync --locked`。 |
| 手動執行 `uv run tradingdev-mcp` 後沒有畫面 | 預設為 stdio，正在等待 MCP 客戶端通訊，不會顯示對話視窗；一般由客戶端自動啟動即可。 |
| 行情下載失敗或資料不足 | 請助手列出資料來源、檢查交易對、K 線頻率與期間，並讀取工作錯誤；確認網路可存取該來源，額外特徵的檔案也已備妥。 |
| 策略無法回測 | 請助手讀取驗證診斷，確認同一版本已通過驗證與試跑；含 `validation` 時改用 walk-forward。 |
| 回測完成，Dashboard 卻找不到結果 | 確認 `TRADINGDEV_WORKSPACE` 與 MCP 的 `--workspace` 相同，且結果有可載入的 `pipeline_result`；最佳化結果改在對話中查詢。 |
| 舊策略顯示 `revision_required` | 請助手讀取舊程式與設定，另存為新版本，再驗證、試跑。原有檔案不會自動遷移。 |
| 升級後舊策略或圖表無法使用 | 舊 `pandas_ta`／`indicator_column` 用法須依策略契約更新；`random_seed` 須放在 YAML 頂層。舊結果可讀範圍取決於保存的產物，完整新版圖表可能需要重新回測。 |

技術指標套件 TA-Lib 隨安裝命令一併安裝。若平台有相容 wheel，不需另外安裝 C 函式庫；
只有必須從原始碼編譯時，才需依[TA-Lib 安裝說明](https://github.com/TA-Lib/ta-lib-python#installation-)
準備底層依賴。

## 相關文件

- [內建策略說明](docs/strategies/README.md)：各策略的規則、參數與資料需求。
- [策略契約](docs/strategy_contract.md)：自行撰寫 Python／YAML、版本與驗證規則。
- [執行結果與產物](docs/run_artifacts.md)：完整指標、執行設定與保存格式。
- [開發指南](docs/development.md)：修改 TradingDev 本身時的環境、品質檢查、測試與 Git／PR 流程。
- [專案架構](ARCHITECTURE.md)：服務分工、MCP 回覆契約與執行流程。
- [代理開發政策](AGENTS.md)：協助開發此專案的代理須遵守的規則。
