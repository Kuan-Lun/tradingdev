# 介面需求

本冊定義外部入口與契約探索；整體範圍見 [SRS](../requirements.md)。

## NFR-API-001 可探索的標準介面

- **需求**：後端透過標準 MCP 提供工具、schema 與可按需取得的契約，維持模型供應商獨立性；客戶端不需讀取 server 原始檔才能取得必要操作資訊。
- **驗收**：未知參數及無效回覆拒絕，預期應用失敗提供穩定診斷；客戶端不能只靠 isError 判定業務成功。契約章節可探索，無效章節明確失敗。CLI 採同步執行且不使用 MCP 對話確認；Dashboard 呈現已保存結果，不把後端通用性當成所有客戶端皆已實測。
- **契約**：[MCP 回覆與探索](../../ARCHITECTURE.md#mcp-回覆契約)、[策略契約入口](../strategy_contract.md#contract-references)、[入口與服務責任](../../ARCHITECTURE.md#系統邊界與模組責任)、[CLI 執行](../../README.md#終端機用法)。
- **驗證**：[真實 MCP 協定](../../tests/integration/test_mcp_protocol.py)、[嚴格輸入邊界](../../tests/mcp/test_strict_input_boundary.py)、[契約探索檢查](../../tests/tooling/test_prompt_contracts.py) 分別驗證協定、輸入及可發現性。
