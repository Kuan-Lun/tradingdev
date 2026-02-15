# GLFT Market-Making Strategy

## 策略概述

基於 **Guéant–Lehalle–Fernandez-Tapia (2013)** 最優做市模型的交易策略。核心思想：根據市場波動率和庫存風險動態調整進出場閾值，在安全時進場交易，並透過庫存懲罰機制自然退出。

與 Safety-First Volume Strategy 的最大區別是：本策略**不需要 ML 模型**，完全基於解析公式計算，運行速度快且可解釋性強。

適用場景：第三方造市服務，需達成指定交易量（如單邊 12.5M USDT/月），同時控制虧損。

## 理論基礎

### 原始 GLFT 模型

GLFT 模型解決了做市商的庫存風險問題，給出最優報價：

- **Reservation price** (庫存調整公允價格):
  ```
  r(t, q) = s(t) - q · γ · σ² · τ
  ```

- **Optimal spread** (最優報價差):
  ```
  δ*(t) = γ · σ² · τ + (2/γ) · ln(1 + γ/κ)
  ```

其中：
- `s(t)` = 中間價 (mid-price)
- `q` = 庫存方向 (+1=多頭, -1=空頭, 0=空倉)
- `γ` = 風險厭惡係數 (越大越保守)
- `σ` = 波動率
- `τ` = 剩餘持倉期 (T - t)
- `κ` = 訂單到達強度

### 信號型適配

由於我們的框架是信號驅動（非限價單），適配方式如下：

1. 使用 **EMA** 作為公允價格 `s(t)` 的估計
2. 計算價格偏離度：`deviation = (close - EMA) / EMA`
3. 計算正規化 half-spread 作為進出場閾值
4. 庫存調整偏離度決定是否退出

```
FLAT ──(|deviation| > half_spread)──► LONG/SHORT
                                          │
                         ┌────────────────┼────────────────┐
                         ▼                ▼                ▼
                   min_holding       adjusted_dev       max_holding
                   未到 → 維持      超過閾值 → 平倉     → 強制平倉
```

### 關鍵行為

- **高波動 → 寬 spread → 減少交易** (自我調節)
- **持有多頭 → reservation price 下移 → 更容易平倉**
- **持有空頭 → reservation price 上移 → 更容易平倉**
- **τ 遞減 → spread 收窄 + 庫存懲罰增大 → 自然退出**

## 參數說明

### Core GLFT

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `gamma` | 0.01 | 風險厭惡係數 (γ)，越大越保守 |
| `kappa` | 1.5 | 訂單到達強度 (κ)，越大 spread 越窄 |
| `ema_window` | 21 | EMA 計算窗口，用於估計公允價格 |

### Volatility

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `vol_window` | 30 | 滾動波動率估計窗口 |
| `vol_type` | "realized" | 波動率類型："realized" (log-return std) 或 "parkinson" (high-low) |

### Holding Management

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `min_holding_bars` | 5 | 最短持倉（5 min @1m）— 符合交易所規則 |
| `max_holding_bars` | 30 | 最長持倉（30 min @1m）— 對應 GLFT 的 horizon T |

### fit() Grid Search

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `gamma_candidates` | [0.001..0.1] | 搜尋最佳 γ |
| `kappa_candidates` | [0.5..5.0] | 搜尋最佳 κ |
| `ema_window_candidates` | [10, 21, 50] | 搜尋最佳 EMA 窗口 |
| `target_metric` | "total_return" | 優化目標指標 |

### Position & Volume

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `position_size_usdt` | 3000.0 | 每筆持倉大小 |
| `monthly_volume_target_usdt` | 12500000 | 月度單邊交易量目標 |
| `fee_rate` | 0.0011 | 單邊手續費 + 滑點 |

## 交易所規則合規

**限制**: ≤3 min 交易占比 < 60%, ≤5 min 交易占比 < 75%

**合規機制**:
- `min_holding_bars >= 5`（@1m timeframe = 5 分鐘）
- 狀態機強制執行：持倉期間內不可提前退出
- 較大的 position_size (3000 USDT) 減少所需交易次數

## YAML 配置範例

```yaml
strategy:
  name: "glft_market_making"
  parameters:
    gamma: 0.01
    kappa: 1.5
    ema_window: 21
    vol_window: 30
    vol_type: "realized"
    min_holding_bars: 5
    max_holding_bars: 30
    gamma_candidates: [0.001, 0.005, 0.01, 0.05, 0.1]
    kappa_candidates: [0.5, 1.0, 1.5, 2.0, 5.0]
    ema_window_candidates: [10, 21, 50]
    target_metric: "total_return"
    position_size_usdt: 3000.0
    monthly_volume_target_usdt: 12500000

backtest:
  # ... (see configs/glft_strategy.yaml)
  signal_as_position: true    # signal=0 → 平倉
  re_entry_after_sl: false    # SL 後不自動重入場
  stop_loss: 0.03             # 3% 緊急 SL
  mode: "volume"
```

## 使用方式

```bash
uv run python -m btc_strategy.main --config configs/glft_strategy.yaml
```

## 與其他策略的比較

| | Safety-First Volume | GLFT Market Making |
|---|---|---|
| 進場機制 | XGBoost P(safe) | 解析 spread 閾值 |
| 方向決定 | SMA 交叉 / ML | 價格偏離 EMA 方向 |
| ML 依賴 | 重度 (XGBoost) | 無 |
| 出場機制 | 風險分數下降 | τ 遞減 + 庫存懲罰 |
| 參數優化 | Lookback + threshold 搜尋 (AUC-ROC) | Grid search (γ, κ, EMA) via backtest |
| 滾動重訓 | 每 N bars 重訓 XGBoost | 不需要 (解析模型) |
| 計算成本 | 高 (ML 訓練/推理) | 低 (EMA + std + for-loop) |
| 理論基礎 | 風險分類 | 最優做市 (GLFT) |
| 波動適應 | 作為特徵輸入 ML | 直接影響 spread 公式 |

## 參考文獻

- Guéant, O., Lehalle, C.-A., & Fernandez-Tapia, J. (2013). *Dealing with the Inventory Risk: A Solution to the Market Making Problem.* Mathematics and Financial Economics, 7(4), 477-507.
