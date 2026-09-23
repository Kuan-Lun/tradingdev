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

Bundled config 使用 `strategy.id`、`strategy.class_name`、`strategy.source_path`
與 `data.requirements` schema。Generated strategies 不放在此目錄；它們由 MCP
lifecycle 管理，每次保存都在
`workspace/generated_strategies/<strategy_id>/revisions/<revision_id>/` 建立
`strategy.py`、`config.yaml` 與 `metadata.json`。每個 revision 分別保存驗證證據與
`draft → validated → runnable → promoted` 狀態，新版本不繼承舊版本的執行資格。
`workspace/configs/<id>.yaml` 僅為舊格式策略的恢復讀取位置；舊檔須明確另存並重新驗證。
