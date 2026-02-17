# GLFT Market-Making Strategy

## 策略概述

基於 **Guéant–Lehalle–Fernandez-Tapia (2013)** 最優做市模型的交易策略。核心思想：根據市場波動率和庫存風險動態調整進出場閾值，在安全時進場交易，並透過利潤目標和止損機制控制風險。

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
- `σ` = 波動率 (per-bar)
- `τ` = 剩餘持倉期 (max_hold - bars_in_pos)
- `κ` = 訂單到達強度

### Pure %-Space 公式

本實作使用純百分比空間的 GLFT 公式，**不依賴資產價格水平**：

```
δ_pct = γ · σ² · τ / 2 + (1/γ) · ln(1 + γ/κ)
```

γ 和 κ 均為無量綱的 %-space 參數。當 BTC 價格從 $40K 變動到 $100K 時，spread 只隨波動率 (σ) 變化，不受價格水平影響。

### 信號型適配

由於我們的框架是信號驅動（非限價單），適配方式如下：

1. 使用 **EMA** 作為公允價格 `s(t)` 的估計
2. 計算價格偏離度：`deviation = (close - EMA) / EMA`
3. 計算正規化 half-spread 作為進場閾值
4. 利潤目標和止損控制出場

```
FLAT ──(|deviation| > half_spread)──► LONG/SHORT
                                          │
                         ┌────────────────┼────────────────┐
                         ▼                ▼                ▼
                   min_holding      profit_target /     max_holding
                   未到 → 維持      SL → 平倉           → 強制平倉
```

### 進場邏輯

1. 計算 GLFT half-spread：`glft_hs = γ·σ²·τ/2 + spread_const`
2. 取 `max(glft_hs, min_entry_edge)` 作為進場閾值
3. `deviation < -threshold` → 做多
4. `deviation > threshold` → 做空
5. 可選動量防護：要求 deviation 正在收斂（價格回歸 EMA）
6. 可選趨勢過濾：只順勢開倉

### 出場邏輯

三個出場條件（優先順序）：

1. **策略級止損** (`strategy_sl`)：deviation 進一步偏離超過閾值時止損
2. **利潤目標** (`profit_target_ratio`)：deviation 回歸到目標水平時獲利了結
   - ratio=1.0 → 等待完全回歸到 EMA (deviation=0)
   - ratio=0.5 → 等待回歸 50%
   - ratio=1.5 → 等待超過 EMA（overshoot）
3. **最大持倉** (`max_holding_bars`)：強制平倉

### 關鍵行為

- **高波動 → 寬 spread → 減少交易** (自我調節)
- **低波動 → 窄 spread → 更多交易** (增加流動性)
- **价格水平無關**：純 %-space 公式確保 spread 只隨 σ 變化
- **止損控制尾部風險**：避免持有到 max_hold 的大額虧損
- **利潤目標鎖定收益**：不再過早出場切割利潤

## 參數說明

### Core GLFT (%-space)

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `gamma` | 500.0 | 風險厭惡係數 (γ)，越大 spread 越寬 |
| `kappa` | 1000.0 | 訂單到達強度 (κ)，越大 spread 越窄 |
| `ema_window` | 21 | EMA 計算窗口，用於估計公允價格 |

### Volatility

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `vol_window` | 30 | 滾動波動率估計窗口（`implied` 時不使用） |
| `vol_type` | "realized" | `"realized"` / `"parkinson"` / `"implied"` |
| `dvol_raw_path` | null | DVOL 原始 CSV 路徑（`implied` 時必填） |
| `dvol_processed_path` | null | DVOL 處理後 Parquet 路徑（`implied` 時必填） |

### Implied Volatility (DVOL)

當 `vol_type: "implied"` 時，策略使用 **Deribit DVOL** 指數作為波動度來源，而非從歷史價格估計。

**DVOL 簡介**：
- Deribit 的 BTC 隱含波動度指數，類似傳統市場的 VIX
- 衡量市場預期的未來 30 天年化波動度（單位：%）
- 由選擇權市場價格反推（forward-looking）
- 公開 API，免 API Key，支援 1 分鐘解析度

**轉換公式**（年化 % → per-bar sigma）：
```
sigma_per_bar = DVOL / 100 / sqrt(525960)
```
其中 `525960 = 365.25 × 24 × 60`（年分鐘數）。

### Multi-Timeframe

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `signal_agg_minutes` | 1 | 信號聚合時間（分鐘）；1=不聚合，5=5 分鐘 EMA |
| `signal_agg_minutes_candidates` | [1] | Grid search 候選值 |

當 `signal_agg_minutes > 1` 時，EMA 在聚合後的 N 分鐘 K 棒上計算，再映射回 1 分鐘解析度（帶 1-period lag 避免前視偏差）。執行仍在 1 分鐘精度。效果：偏離度更大、信號更穩定。

### Entry

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `min_entry_edge` | 0.0012 | 最低進場偏離閾值 |
| `momentum_guard` | true | 動量防護：只在 deviation 收斂時進場 |
| `trend_ema_window` | 0 | 趨勢過濾慢 EMA；0=停用 |

### Exit

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `profit_target_ratio` | 1.0 | 利潤目標比率 (1.0=完全回歸到 EMA) |
| `strategy_sl` | 0.005 | 策略級止損閾值（0=停用） |

### Holding Management

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `min_holding_bars` | 5 | 最短持倉（5 min @1m）— 符合交易所規則 |
| `max_holding_bars` | 30 | 最長持倉（30 min @1m）— 對應 GLFT 的 horizon T |

### fit() Grid Search

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `gamma_candidates` | [0, 200, 500, 1000] | 搜尋最佳 γ |
| `kappa_candidates` | [500, 1000, 5000] | 搜尋最佳 κ |
| `ema_window_candidates` | [2, 3, 5] | 搜尋最佳 EMA 窗口 |
| `profit_target_ratio_candidates` | [0.5, 0.75, 1.0] | 搜尋最佳利潤目標 |
| `target_metric` | "total_volume_usdt" | 優化目標指標 |

### Position & Volume

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `position_size_usdt` | 3000.0 | 每筆持倉大小 |
| `monthly_volume_target_usdt` | 12500000 | 月度單邊交易量目標 |
| `fee_rate` | 0.0006 | 單邊手續費 |

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
    gamma: 500
    kappa: 1000
    ema_window: 21
    vol_window: 30
    vol_type: "implied"
    dvol_processed_path: "data/processed/btc_dvol_1m_2024_2025.parquet"
    min_holding_bars: 5
    max_holding_bars: 30
    gamma_candidates: [0, 200, 500, 1000]
    kappa_candidates: [500, 1000, 5000]
    ema_window_candidates: [2, 3, 5]
    max_holding_bars_candidates: [6, 8, 13]
    target_metric: "total_volume_usdt"
    position_size_usdt: 3000.0
    monthly_volume_target_usdt: 12500000
    fee_rate: 0.0006
    min_entry_edge: 0.0015
    min_entry_edge_candidates: [0.0008, 0.001, 0.0012]
    profit_target_ratio: 1.0
    profit_target_ratio_candidates: [0.5, 0.75, 1.0]
    strategy_sl: 0.003
    momentum_guard: false
    trend_ema_window: 0
    trend_ema_candidates: [0, 200]
    min_annual_return: -0.18

backtest:
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
| 出場機制 | 風險分數下降 | 利潤目標 + 止損 + 最大持倉 |
| 參數優化 | Lookback + threshold 搜尋 (AUC-ROC) | Grid search (γ, κ, EMA, pt_ratio) via backtest |
| 滾動重訓 | 每 N bars 重訓 XGBoost | 不需要 (解析模型) |
| 計算成本 | 高 (ML 訓練/推理) | 低 (EMA + std + for-loop) |
| 理論基礎 | 風險分類 | 最優做市 (GLFT) |
| 波動適應 | 作為特徵輸入 ML | 直接影響 spread 公式 |
| 價格獨立 | 否（需 ML 重訓） | 是（純 %-space 公式） |

## 參考文獻

- Guéant, O., Lehalle, C.-A., & Fernandez-Tapia, J. (2013). *Dealing with the Inventory Risk: A Solution to the Market Making Problem.* Mathematics and Financial Economics, 7(4), 477-507.
