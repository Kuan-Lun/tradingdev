# Safety-First Volume Strategy

## 策略概述

以**最小虧損**滿足**月度交易量目標**的策略。核心思想：只在安全的市場環境下進場交易，避開高風險時段。

適用場景：第三方造市服務，需達成指定交易量（如單邊 12.5M USDT/月），同時控制虧損。

## 三層架構

### Layer 1: Risk Gate (XGBoost)

判斷「現在進場是否安全」。

- **模型**: XGBoost 二元分類
- **Target 定義**:
  - safe=1: 未來 `target_holding_bars` 根 bar 內，**最佳方向**的 round-trip P&L > `-max_acceptable_loss_pct`
  - safe=0: 即使選對方向也會虧損超過門檻
- **決策**: `P(safe) >= risk_threshold` → 允許進場
- **特徵**: 波動率（多 horizon）、ATR、布林帶寬度、ADX、成交量異常、K 線結構、收益率偏度/峰度

### Layer 2: Direction (SMA Crossover)

Risk Gate 放行後決定做多或做空。

- **預設**: SMA 快線 > 慢線 → 做多，反之做空
- **可選**: 切換為 ML 方向模型（`use_ml_direction: true`）

### Layer 3: Holding Manager (狀態機)

管理持倉生命週期，確保符合交易所規則。

```
FLAT ──(safe + direction)──► LONG/SHORT
                                 │
                    ┌────────────┼────────────┐
                    ▼            ▼            ▼
              min_holding   risk < thr    max_holding
              未到 → 維持   → 平倉        → 平倉
                    │
                    ▼
            direction 反轉
            → 反轉持倉 (2× volume)
```

## 參數說明

### Risk Gate

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `risk_model` | (see YAML) | XGBoost 超參數 |
| `risk_threshold` | 0.5 | P(safe) 需超過此值才允許進場 |
| `risk_threshold_candidates` | [0.3..0.7] | fit() 時自動搜尋最佳 threshold |
| `target_holding_bars` | 7 | 計算 safe target 時的假設持倉期 |
| `max_acceptable_loss_pct` | 0.003 | 超過 0.3% 虧損 → 標記為 unsafe |
| `fee_rate` | 0.0011 | 單邊手續費 + 滑點 |

### Direction

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `use_ml_direction` | false | true: 使用 XGBoost 方向模型 |
| `sma_fast` | 5 | SMA 快線週期 |
| `sma_slow` | 20 | SMA 慢線週期 |

### Holding Management

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `min_holding_bars` | 5 | 最短持倉（5 min @1m）— 符合交易所規則 |
| `max_holding_bars` | 30 | 最長持倉（30 min @1m） |

### Training

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `lookback_candidates` | [360, 720, 1440] | 搜尋最佳特徵 lookback |
| `retrain_interval` | 720 | 每 720 根 bar（12h @1m）重訓模型 |
| `validation_ratio` | 0.2 | 驗證集比例 |

### Position & Volume

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `position_size` | 3000.0 | 每筆持倉大小 |
| `monthly_volume_target` | 12500000 | 月度單邊交易量目標 |

## 交易所規則合規

**限制**: ≤3 min 交易占比 < 60%, ≤5 min 交易占比 < 75%

**合規機制**:
- `min_holding_bars >= 5`（@1m timeframe = 5 分鐘）
- 狀態機強制執行：持倉期間內不可提前退出（僅緊急 SL 例外）
- 較大的 position_size (3000 USDT) 減少所需交易次數

## YAML 配置範例

```yaml
strategy:
  name: "safety_first_volume"
  parameters:
    risk_model:
      n_estimators: 300
      max_depth: 4
      learning_rate: 0.05
    risk_threshold: 0.5
    risk_threshold_candidates: [0.3, 0.4, 0.5, 0.6, 0.7]
    target_holding_bars: 7
    max_acceptable_loss_pct: 0.003
    fee_rate: 0.0011
    use_ml_direction: false
    sma_fast: 5
    sma_slow: 20
    min_holding_bars: 5
    max_holding_bars: 30
    lookback_candidates: [360, 720, 1440]
    retrain_interval: 720
    position_size: 3000.0
    monthly_volume_target: 12500000

backtest:
  # ... (see src/tradingdev/domain/strategies/bundled/safety_volume_strategy/config.yaml)
  signal_as_position: true    # signal=0 → 平倉
  re_entry_after_sl: false    # SL 後不自動重入場
  stop_loss: 0.03             # 3% 緊急 SL
```

## 使用方式

```bash
uv run python -m tradingdev --config src/tradingdev/domain/strategies/bundled/safety_volume_strategy/config.yaml --walk-forward
```

## 與 XGBoost Direction Strategy 的比較

| | XGBoost Direction | Safety-First Volume |
|---|---|---|
| 目標 | 預測方向 | 預測安全性 |
| 進場頻率 | 每根 bar | 安全時才進場 |
| 持倉大小 | 200 USDT | 3,000 USDT |
| 持倉時間 | < 1 min | 5-30 min |
| SL/TP | 緊密（0.4-0.5%） | 寬鬆（3% 緊急） |
| signal=0 | 不操作 | 平倉 |
| 交易所合規 | 不符合 | 符合 |
