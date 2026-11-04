# ML 訊號研究需求

本冊對應 UC-04；整體範圍見 [SRS](../requirements.md)。
此為獨立同步研究介面，不建立 execution plan 或背景回測工作，不自動把模型轉為策略。

## FR-ML-001 能力探索與固定資料集

- **需求**：依探索所得的已登錄特徵、模型及驗證方式建置研究；資料集只讀可重用的既有行情，保存不可變內容版本與來源。
- **驗收**：來源、市場、代號、頻率與區間一致；快取缺少、尚不可重用或選取期間為空時失敗，不隱式下載。暖機及無法形成標籤的列被明確計數，未來標籤不可冒充當時可得特徵；成功不保證來源 K 棒完整。
- **契約**：[資料集與模型輸入](../ml_research.md#dataset-and-model-inputs)、[不可變資料集](../ml_research.md#immutable-datasets)。
- **驗證**：[資料集完整性](../../tests/app/test_ml_dataset_integrity.py)、[時間完整性](../../tests/app/test_ml_temporal_integrity.py) 檢查來源、內容及時間邊界。

## FR-ML-002 訓練、驗證與預測

- **需求**：訓練固定模型參數及種子，保存模型與特徵身分；分折驗證與最後使用全部訓練列擬合的模型分開解讀。預測核對特徵順序與模型內容。
- **驗收**：不把分折分數當作最終模型的樣本外證明；無效資料、來源綁定或特徵不符須拒絕。完整預測與回覆預覽分開，預覽分頁不裁減實際評估的訊號。模型檔案需可信，身分檢查不使 pickle 成為安全格式。
- **契約**：[分折與訊號解讀](../ml_research.md#fold-validation-and-signal-interpretation)、[模型保存](../ml_research.md#trained-models)。
- **驗證**：[訓練驗證](../../tests/app/test_ml_training_validation.py)、[預測 service](../../tests/app/test_ml_predict.py) 核對分折與預測邊界。

## FR-ML-003 訊號與手設規則評估

- **需求**：模型訊號及手設規則可在明示資料與持有期間下評估，核對完整資料對齊，保存方法家族、參數、訊號指紋與評估條件。
- **驗收**：評估包含無訊號、暖機及不重疊持有的限制；簡化無成本統計不宣稱等同交易回測或具有成交／權益證據。看過資料後選門檻保留樣本內選擇偏誤；未保存的序列不事後猜測補算。
- **契約**：[分折與訊號解讀](../ml_research.md#fold-validation-and-signal-interpretation)、[已記錄訊號評估](../ml_research.md#recorded-signal-evaluations)。
- **驗證**：[規則評估](../../tests/app/test_ml_rule_evaluation.py)、[訊號收益](../../tests/domain/ml/test_simulated_returns.py) 核對持有區間、評估與無成本口徑。

## FR-ML-004 排名與比較條件

- **需求**：依指標定義的方向排名，保留同分名次、不可用原因與比較條件；只有明確選擇時才跨參數軸比較。
- **驗收**：不可用候選不被轉為零或默默省略；不同條件的排名須揭露比較限制，尤其跨 horizon 的來源行情一致性未受保證。Run 與 signal 的共用查詢入口保持各自的資料與評估語意。
- **契約**：[訊號排名](../ml_research.md#recorded-signal-evaluations)、[保存結果排名](../run_artifacts.md#ranking-saved-run-scopes)。
- **驗證**：[ML 排名 service](../../tests/app/test_ml_leaderboard.py)、[MCP 共用排名](../../tests/mcp/test_ml_leaderboard_runs.py) 核對排序、不可用值及資料來源分派。
