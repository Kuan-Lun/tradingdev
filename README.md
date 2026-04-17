# 量化交易回測 MCP Server

本專案是一個 **MCP (Model Context Protocol) Server**，讓使用者可以在任何支援 MCP 的聊天視窗（Claude Desktop、claude.ai 等）中，用自然語言描述策略想法，**由 LLM 直接撰寫全新的策略程式碼與配置、存檔、並觸發回測**。策略不受限於任何預設清單——LLM 會根據使用者描述，用 pandas 自行實作指標與信號邏輯，產生的 Python 檔與 YAML 配置會寫入專案內，隨後可立即回測、迭代優化。

底層為基於 `vectorbt` 的向量化回測框架，支援多種標的（如 BTC/USDT 永續合約等）。

## Demo

[![Demo 影片](https://img.shields.io/badge/Demo-%E8%A7%80%E7%9C%8B%E5%BD%B1%E7%89%87-blue)](https://url.klwang.tw/qbdemo)

影片示範典型的 vibe coding 流程：

1. 使用者用提示詞向 LLM 描述策略構想（例如「我想要一個結合布林通道與成交量突破的做多策略」）
2. LLM 回傳策略設計建議（指標選擇、進出場條件、參數、風險控制）
3. 使用者確認或修改後，LLM 自動產生 Python 策略檔與 YAML 配置，寫入專案
4. LLM 觸發回測，回傳 Sharpe、Max Drawdown、Win Rate 等績效指標
5. 使用者根據結果與 LLM 繼續迭代：調整參數、加條件、換指標、甚至整個重寫

> **新開發的策略會寫入這兩個路徑**（透過 MCP 的 `save_strategy` 工具）：
>
> - [strategies/](strategies/)：Python 策略實作（新檔案，繼承 `BaseStrategy`）
> - [configs/](configs/)：YAML 策略與回測配置（新檔案）
>
> 既有的策略只是已開發過的範例，並非選單——LLM 可以自由生出全新名稱的策略，也可以在使用者同意時基於既有策略修改。

## 環境需求

- Python 3.12 ~ 3.13
- [uv](https://docs.astral.sh/uv/) (Python 套件管理工具)
- [cloudflared](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/)（若要讓 claude.ai 等雲端 LLM 連線到本機 MCP Server）

### 安裝 uv

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# 或透過 Homebrew
brew install uv
```

## 快速開始

```bash
# 1. Clone 專案
git clone <repo-url>
cd tradingdev.clone

# 2. 安裝依賴並建立虛擬環境
uv sync

# 3. 驗證安裝
uv run python -c "import quant_backtest; print('OK')"
```

## 啟動 MCP Server

### 模式一：本機 stdio（Claude Desktop）

將以下設定加入 `~/Library/Application Support/Claude/claude_desktop_config.json`：

```json
{
  "mcpServers": {
    "quant-backtest": {
      "command": "uv",
      "args": ["run", "python", "mcp_server/server.py"],
      "cwd": "/absolute/path/to/tradingdev.clone"
    }
  }
}
```

重啟 Claude Desktop 即可使用。

### 模式二：HTTP + Cloudflare Tunnel（claude.ai 等網頁端）

```bash
# Terminal 1：啟動 HTTP MCP Server
uv run python mcp_server/server.py --web --port 8000

# Terminal 2：開啟公開 tunnel，取得 https://xxx.trycloudflare.com
cloudflared tunnel --url http://localhost:8000
```

將 cloudflared 輸出的 HTTPS URL 加入 claude.ai 的 MCP connector 設定，即可在網頁聊天中使用。

## MCP 工具一覽

Server 透過 FastMCP 暴露以下工具（詳見 [mcp_server/server.py](mcp_server/server.py)）：

### 策略開發

| 工具 | 用途 |
| ---- | ---- |
| `get_strategy_template` | 取得 `BaseStrategy` ABC 原始碼、範例程式、YAML 樣板與 API 參考——LLM 寫新策略前必呼叫 |
| `save_strategy` | 把 LLM 產生的 Python + YAML 寫入 [strategies/](strategies/) 與 [configs/](configs/)，支援新建或覆寫同名檔 |
| `list_strategies` | （可選）列出專案內已存在的策略，讓 LLM 判斷是否有可參考或修改的既有實作 |
| `get_strategy` | （可選）讀取指定策略的原始碼與配置，作為修改基底 |

### 資料與回測

| 工具 | 用途 |
| ---- | ---- |
| `list_available_data` | 列出 `data/processed/` 中已快取的 OHLCV 資料 |
| `start_backtest` | 非同步觸發回測，立即回傳 `job_id`（不阻塞聊天） |
| `start_optimization` | 非同步觸發參數 grid search |
| `get_job_status` | 查詢回測 / 優化進度或最終績效指標 |
| `list_jobs` | 列出所有 job |

## 典型對話流程

1. 使用者描述策略構想（完全原創亦可，例如「結合 ATR 的動態停損 + EMA 斜率過濾的趨勢跟隨」）
2. LLM 呼叫 `get_strategy_template` 取得基底類別、範例與 API 參考
3. **（可選）** LLM 呼叫 `list_strategies`，判斷是否剛好有類似的既有策略可以修改；若無或使用者偏好新建，LLM 直接用 pandas 撰寫全新程式碼
4. LLM 呼叫 `save_strategy`，在 [strategies/](strategies/) 與 [configs/](configs/) 下**建立全新檔案**（檔名由 LLM 依策略命名）
5. LLM 向使用者說明實作邏輯，確認後呼叫 `start_backtest`
6. 使用者詢問進度 → LLM 呼叫 `get_job_status` 取得績效（Sharpe、Max Drawdown、Win Rate…）
7. 根據結果迭代：調整參數、改信號邏輯、加風控條件、或要求 LLM 從頭改寫——每次 `save_strategy` 可覆寫同名檔或另存為新策略

## 已開發過的策略（參考用）

下列為本專案內附的策略實作範例——非 MCP 的「可選清單」，而是過去 vibe coding 累積的成果，可作為 LLM 學習或修改的基底：

| 策略 | 配置檔 | 說明 |
| ---- | ------ | ---- |
| KD Crossover | [configs/kd_strategy.yaml](configs/kd_strategy.yaml) | KD 隨機指標黃金交叉 / 死亡交叉 |
| XGBoost Direction | [configs/xgboost_strategy.yaml](configs/xgboost_strategy.yaml) | XGBoost 方向預測 |
| Safety Volume | [configs/safety_volume_strategy.yaml](configs/safety_volume_strategy.yaml) | 安全優先量化策略（風控閘門 + 方向預測） |
| Quantile Regime | [configs/quantile_strategy.yaml](configs/quantile_strategy.yaml) | 四分類 regime detection |
| GLFT | [configs/glft_strategy.yaml](configs/glft_strategy.yaml) | GLFT 最優做市模型 |
| GLFT + ML | [configs/glft_ml_strategy.yaml](configs/glft_ml_strategy.yaml) | GLFT + AutoGluon ML 方向預測 |

## CLI 直接回測（非 MCP 用法）

若不透過 LLM，也可直接用 CLI 執行回測：

```bash
uv run python -m quant_backtest.main --config configs/<strategy>.yaml
```

回測結果會快取至 `data/cache/`，可用 Streamlit Dashboard 檢視：

```bash
uv run streamlit run src/quant_backtest/dashboard/app.py -- --config configs/<strategy>.yaml
```

## 相關文件

- [CLAUDE.md](CLAUDE.md)：Claude Code 協作指引與編碼規範
- [ARCHITECTURE.md](ARCHITECTURE.md)：模組關係與類別圖
- [docs/strategies/](docs/strategies/)：各內建策略的詳細說明

## 常見問題

- **editable install 失效 / `ModuleNotFoundError`**：請確認 uv ≥ 0.10，必要時 `uv self update && rm -rf .venv && uv sync`。
- **macOS 上 `.pth` 被忽略**：Python 3.13 的 `site` 模組會跳過帶 `UF_HIDDEN` 屬性的 `.pth`，執行 `uv pip install --reinstall -e .` 可修復。
