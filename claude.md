# BTC/USDT 合約交易策略回測專案

## 專案概述

本專案用於開發與回測 BTC/USDT 永續合約交易策略。目前範圍僅限於歷史回測，不包含實盤交易。
開發模式以 vibe coding 為主，由 Claude 撰寫程式碼，需嚴格遵守本文件規範。

## 技術棧

- **語言**: Python 3.11+
- **套件管理**: uv（透過 pyproject.toml 管理依賴）
- **回測框架**: vectorbt
- **資料處理**: pandas, numpy
- **技術指標**: pandas-ta 或 ta-lib
- **機器學習**: scikit-learn, xgboost
- **資料驗證**: pydantic
- **資料爬取**: ccxt（Binance 公開 API）、httpx（Data Vision 批次下載）
- **Linter/Formatter**: ruff
- **型別檢查**: mypy（全局 strict 模式）
- **測試**: pytest
- **配置格式**: YAML

## 專案結構

```
btc-usdt-strategy/
├── pyproject.toml              # uv 專案配置、依賴管理
├── claude.md                   # Claude 協作指引
├── mypy.ini                    # mypy strict 配置
├── ruff.toml                   # Ruff 配置
├── configs/                    # 策略參數與回測配置 (YAML)
├── data/
│   ├── raw/                    # 原始爬取資料 (CSV/ZIP)
│   └── processed/              # 清洗對齊後的回測用資料 (Parquet)
├── src/
│   └── btc_strategy/           # 主套件 (src layout)
│       ├── crawlers/           # 資料爬取模組
│       │   ├── base.py         # ABC: BaseCrawler
│       │   ├── binance_api.py  # Binance 公開 API 爬取
│       │   └── binance_vision.py # Data Vision 批次下載
│       ├── data/               # 資料讀取與處理
│       │   ├── schemas.py      # Pydantic 資料模型 (OHLCV 等)
│       │   ├── loader.py       # 讀取本地 CSV/Parquet
│       │   └── processor.py    # 清洗、時區對齊、缺值處理
│       ├── indicators/         # 技術指標模組
│       │   └── base.py         # ABC: BaseIndicator
│       ├── strategies/         # 交易策略模組
│       │   └── base.py         # ABC: BaseStrategy
│       ├── ml/                 # 機器學習模型
│       │   └── base.py         # ABC: BaseModel
│       ├── backtest/           # 回測引擎
│       │   ├── engine.py       # Vectorbt 回測執行器
│       │   └── metrics.py      # 績效指標計算
│       └── utils/              # 共用工具
│           ├── logger.py       # Logging 配置
│           └── config.py       # YAML 配置載入
├── notebooks/                  # 探索性分析 Jupyter notebooks
└── tests/                      # 單元測試 (鏡像 src 結構)
    ├── test_crawlers/
    ├── test_data/
    ├── test_indicators/
    ├── test_strategies/
    ├── test_ml/
    └── test_backtest/
```

## 編碼規範

### SOLID 原則（強制）

- **SRP**: 每個類別與模組只負責單一職責
- **OCP**: 新增策略/指標/爬蟲時，擴展新類別而非修改既有程式碼
- **LSP**: 子類別必須可完全替換父類別使用
- **ISP**: 介面保持精簡，不強迫實作不需要的方法
- **DIP**: 依賴抽象介面（ABC），不依賴具體實作

### Python 風格

- 全面使用 **type hints**，mypy strict 模式下必須通過檢查
- 命名規範：snake_case（變數/函式）、PascalCase（類別）、UPPER_SNAKE_CASE（常數）
- 每個核心模組透過 **ABC (Abstract Base Class)** 定義介面
- 使用 **Pydantic** 定義資料結構與驗證配置參數
- 策略參數集中於 `configs/` 下的 **YAML 檔案**，禁止寫死在程式碼中
- 使用 Python 標準 **logging** 模組，禁止使用 print 進行除錯
- 遵循 ruff 預設規則進行格式化與 lint 檢查

## 資料規範

### K 線欄位定義

| 欄位 | 型別 | 說明 |
|------|------|------|
| timestamp | datetime (UTC) | K 線開盤時間，統一使用 UTC 時區 |
| open | float | 開盤價 |
| high | float | 最高價 |
| low | float | 最低價 |
| close | float | 收盤價 |
| volume | float | 成交量 |

### 資料來源

- **Binance Data Vision** (`data.binance.vision`): 批次下載大量歷史 K 線資料，免註冊
- **Binance 公開 API** (透過 ccxt): 補齊近期資料，免 API Key
- 支援時間粒度：最小 1 分鐘 (1m)
- 原始資料存放 `data/raw/`，清洗後輸出至 `data/processed/` 為 Parquet 格式

### 爬蟲設計

- 所有爬蟲繼承 `BaseCrawler` 抽象類別
- 預留擴展性：未來可新增 Funding Rate、訂單簿等資料類型的爬蟲

## 策略開發規範

### 策略介面

所有策略必須繼承 `BaseStrategy`，實作以下方法：
- `generate_signals()`: 產生進出場信號
- `get_parameters()`: 回傳策略參數定義

### 信號定義

- 做多信號: `1`
- 做空信號: `-1`
- 無信號: `0`

### 合約交易特有考量

- 回測時必須設定手續費（maker/taker fee）
- 必須設定滑點模擬
- 考量合約特有風險：強制平倉、資金費率（未來擴展）

## 回測規範

### 使用 Vectorbt

- 透過 vectorbt 執行向量化回測
- 績效指標至少包含：Sharpe Ratio、Max Drawdown、Win Rate、Profit Factor、Total Return
- 回測結果需可重現（固定隨機種子、記錄資料集版本與時間範圍）

### 避免常見偏差

- **禁止前視偏差 (Look-Ahead Bias)**: 信號計算只能使用當前及歷史資料
- **樣本外測試**: ML 策略必須劃分 train/validation/test 集
- **過擬合警覺**: 參數優化後須進行 walk-forward 或 cross-validation 驗證

## Git Flow

本專案以 vibe coding 模式開發，每次 session 的成果需清楚可辨識。

### 分支策略

- **main**: 只包含可運行的完整成果，禁止直接 push
- **feature/\<name\>**: 每次 vibe coding session 在 feature 分支上開發
  - 命名範例：`feature/kd-strategy`、`feature/add-macd-indicator`
  - 可拆分多個 commit（例如按 phase 拆分），保留開發歷程

### 合併規範

- 使用 **merge commit**（`git merge --no-ff`）合併回 main
- merge commit 訊息需清楚描述該次 session 的完整成果
- 格式：`feat: <一句話描述整體成果>`
- 這樣 `git log --first-parent main` 可看到每次 vibe coding session 的完整成果

### Tag 規範

- 重要里程碑打 tag，格式為 `v<major>.<minor>.<patch>`
- 例如：`v0.0.0`（專案初始化）、`v0.1.0`（第一個可運行策略）

### Commit 訊息格式

分支上的 commit 使用以下前綴：
- `feat:` 新功能
- `fix:` 修復
- `refactor:` 重構
- `test:` 測試
- `docs:` 文件
- `chore:` 雜務（設定檔、依賴等）

## 架構文件維護

- 當新增、刪除或修改類別、模組、介面（ABC）、Pydantic 模型時，須同步更新 `ARCHITECTURE.md` 中對應的 Mermaid 圖表
- 需檢查的圖表：Class Diagram、Pipeline Sequence Diagram、Module Dependency Graph
- 若僅修改函式內部邏輯而未改變類別/模組結構，則無需更新

## 策略文件維護

- 當新增策略時，須在 `docs/strategies/` 下新增對應的 `.md` 說明文件，並更新 `docs/strategies/README.md` 索引
- 當修改既有策略的介面、參數、信號邏輯或配置格式時，須同步更新對應的策略文件
- 策略文件應包含：策略原理、參數說明、信號邏輯、對應的 YAML 配置範例、使用方式
- 若僅修改內部實作細節而未改變外部行為（參數、信號、配置），則無需更新

## 工具鏈指令

```bash
# 套件管理
uv sync                      # 安裝依賴
uv add <package>             # 新增套件
uv run python <script>       # 執行腳本

# 程式碼品質
uv run ruff check .          # Lint 檢查
uv run ruff format .         # 格式化
uv run mypy src/             # 型別檢查

# 測試
uv run pytest tests/         # 執行全部測試
uv run pytest tests/ -v      # 詳細輸出
```
