# 量化交易策略回測框架

量化交易策略的開發與歷史回測框架，支援多種資產類型。

## 環境需求

- Python 3.12 ~ 3.13
- [uv](https://docs.astral.sh/uv/) (Python 套件管理工具)

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
cd quant-backtest

# 2. 安裝依賴並建立虛擬環境（含 editable install）
uv sync

# 3. 驗證安裝
uv run python -c "import quant_backtest; print('OK')"
```

> **注意**：請使用 uv v0.10 以上版本。舊版 uv 建立的 venv 在 Python 3.13 下可能無法正確處理 editable install。
> 如遇到 `ModuleNotFoundError`，請執行 `uv self update` 升級後重新 `rm -rf .venv && uv sync`。
>
> **macOS 使用者**：Python 3.13 的 `site` 模組會跳過帶有 macOS 隱藏屬性（`UF_HIDDEN`）的 `.pth` 檔案。
> 若 editable install 突然失效，可執行 `uv pip install --reinstall -e .` 重新產生 `.pth` 檔案來修復。

## 執行回測

```bash
uv run python -m quant_backtest.main --config configs/<strategy>.yaml
```

回測完成後，結果會自動快取至 `data/cache/`，供 Dashboard 讀取。

### 可用策略

| 策略 | 配置檔 | 說明 |
|------|--------|------|
| KD Crossover | `configs/kd_strategy.yaml` | KD 隨機指標黃金交叉/死亡交叉 |
| XGBoost Direction | `configs/xgboost_strategy.yaml` | XGBoost 方向預測 |
| Safety Volume | `configs/safety_volume_strategy.yaml` | 安全優先量化策略（風控閘門 + 方向預測） |
| GLFT | `configs/glft_strategy.yaml` | GLFT 最優做市模型 |
| GLFT + ML | `configs/glft_ml_strategy.yaml` | GLFT + AutoGluon ML 方向預測 |

## Dashboard UI

回測完成後，透過 Streamlit 啟動視覺化介面：

```bash
uv run streamlit run src/quant_backtest/dashboard/app.py -- --config configs/<strategy>.yaml
```

## 使用 Claude Code 新增策略

本專案以 vibe coding 模式開發，透過 Claude Code 對話即可完成策略新增。以下是建議的 prompt 範例：

### 新增策略

> 幫我新增一個 RSI 均值回歸策略：RSI 低於 30 做多、高於 70 做空。
> 參數包含 rsi_period、overbought、oversold，需支援 grid search 優化。

Claude Code 會自動處理：策略類別、Pydantic config、registry 註冊、YAML 配置檔、策略文件、ARCHITECTURE.md 更新。

### 執行回測

> 幫我用 kd_strategy 跑 2024 全年的 1h 回測

### 查看結果

> 幫我啟動 dashboard 看 kd_strategy 的回測結果

### 加入 Walk-Forward 驗證

> 幫我在 kd_strategy 的配置中加入 walk-forward 驗證，2024 當訓練集、2025 當測試集
