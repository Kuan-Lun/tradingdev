# 策略文件索引

本目錄包含 bundled strategies 的詳細說明文件。對應程式碼與 config 位於
`src/tradingdev/domain/strategies/bundled/<strategy>/`。

| 策略 | 檔案 | 說明 |
|------|------|------|
| KD Stochastic 交叉 | [kd_strategy.md](kd_strategy.md) | KD 隨機指標金叉/死叉 + 超買超賣過濾 |
| Safety-First Volume | [safety_volume_strategy.md](safety_volume_strategy.md) | 風險門控 + SMA 方向，最小虧損達成交易量目標 |
| GLFT Market Making | [glft_strategy.md](glft_strategy.md) | GLFT 最優做市模型，解析型 spread-based 進出場（支援 DVOL implied vol） |
| GLFT + ML Direction | [glft_ml_strategy.md](glft_ml_strategy.md) | GLFT 做市 + AutoGluon ML 方向預測，limit order market making |
| XGBoost Direction | [xgboost_strategy.md](xgboost_strategy.md) | XGBoost 方向預測 + rolling retrain，volume mode 交易量策略 |
| Quantile Regime Volume | [quantile_strategy.md](quantile_strategy.md) | XGBoost regime classifier + DVOL/funding rate feature requirements |

Generated strategies 不放在此目錄；它們由 MCP lifecycle 管理，位於
`workspace/generated_strategies/` 與 `workspace/configs/`。
