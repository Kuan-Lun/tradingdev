# 資料需求

本冊支援 UC-01 至 UC-04；整體範圍見 [SRS](../requirements.md)。

## FR-DATA-001 來源探索與資料可用性

- **需求**：研究者須能探索來源與需求，依所選市場、原始代號、頻率及期間取得或檢查行情；資料可用性檢查、快取清單與明示 cached-only 讀取不隱式下載資料。
- **驗收**：不同來源及市場不混用價格；無資料、無法取得與資料缺口明確區分。未來區間拒絕；成功查詢或可重用快取不等於 K 棒完整，也不保證外部來源永不修正。
- **契約**：[行情快取](../run_artifacts.md#market-data-cache)。
- **驗證**：[資料 service](../../tests/app/test_data_service.py)、[來源管線](../../tests/integration/test_data_source_pipeline.py) 以替代外部來源檢查查詢與讀取邊界，不證明即時來源可用。

## FR-DATA-002 查詢證據與快取保存

- **需求**：快取以來源、市場、原始代號及頻率識別，依成功查詢時間與覆蓋範圍判斷重用；更新發布新版本，失敗保留既有資料。
- **驗收**：近期區間依正式政策重新查詢；未記錄來源的舊快取不參與新研究。更新失敗不破壞已發布版本，先前取得的版本路徑仍可讀；不宣稱行情版本會自動淘汰。
- **契約**：[行情快取](../run_artifacts.md#market-data-cache)、[DVOL 查詢](../run_artifacts.md#automatic-dvol-queries)。
- **驗證**：[快取身分](../../tests/app/test_data_cache_identity.py)、[DVOL 快取](../../tests/app/test_dvol_cache.py) 核對重用、區間及失敗保存。

## FR-DATA-003 有界取樣與特徵對齊

- **需求**：獨立行情取樣須限制資料處理量，與正式執行準備分開；特徵依時間對齊，只使用當時或較早的有效觀察，不以未來值補開頭缺口。
- **驗收**：取樣不能先完整載入再截斷；下載、解壓及列數超限明確失敗，少量特徵取樣不發布為完整研究查詢證據。缺值保持未知，取樣行數上限不宣稱網路流量同比下降。
- **契約**：[有界歷史取樣](../run_artifacts.md#bounded-historical-sampling)。
- **驗證**：[取樣邊界](../../tests/app/test_sample_data_boundaries.py)、[取樣特徵](../../tests/app/test_sample_features.py) 檢查容量與對齊；外部來源使用替代回覆。
