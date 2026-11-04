# 共通品質需求

本冊適用 UC-01 至 UC-05；整體範圍見 [SRS](../requirements.md)。

## NFR-SAFE-001 程式執行與部署邊界

- **需求**：生成 Python 的靜態限制、受監督執行及 HTTP 入口保護須明確區分；使用者可得知會執行程式的時機及未提供的隔離／驗證能力。
- **驗收**：不把 workspace、靜態檢查或使用者同意稱為完整沙箱。MCP HTTP Host／Origin 依允許清單設定檢查，請求本文依固定容量上限拒絕超量；允許清單不冒充登入驗證；跨設備部署由操作者提供適當驗證與存取控制。平台支援與 POSIX 程序限制如實列明。
- **契約**：[策略安全模型](../strategy_contract.md#security-model)、[連線與部署](../../README.md#2-啟動-mcp-server-並連接助手)。
- **驗證**：[HTTP 入口保護](../../tests/mcp/test_http_transport_security.py)、[真實 preflight worker](../../tests/integration/test_preflight_workers.py) 驗證適用邊界；不證明生成程式的全面安全。

## NFR-OPS-001 有限資源與失敗可觀察性

- **需求**：可成長的清單及輸出使用分頁、明確容量或 artifact；受監督的驗證與準備具期限，背景工作可取消並查詢清理結果。
- **驗收**：超量明確失敗或提供續讀，不默默截斷重要證據。成功、例外與逾時均核對所屬程序停止後清理臨時資料；無法核對時回報並保留診斷。保留使用者成果；外部強殺 supervisor 或自行脫離 session 的後代不在程序保證內。
- **契約**：[程序控制與清理](../../ARCHITECTURE.md#程序控制與清理)、[策略驗證限制](../strategy_contract.md#security-model)、[執行準備](../execution_plans.md#execution-plans-and-confirmation-documents)、[工作控制與清理狀態](../run_artifacts.md#job-state-and-worker-identity)、[探索分頁](../run_artifacts.md#discovery-pages)。
- **驗證**：[preflight 臨時檔案](../../tests/integration/test_preflight_temporary_files.py)、[真實受監督 worker](../../tests/integration/test_managed_workers.py)、[分頁](../../tests/app/test_pagination.py) 檢查清理、狀態與容量。

## NFR-REPRO-001 有界的重現性承諾

- **需求**：執行保存有效設定、版本、資料指紋與適用的隨機種子，讓研究可追溯；結果查詢及報告不因當前預設值改變而重新計算。
- **驗收**：相同受控條件下的種子使用可驗證；策略執行與獨立 ML 訓練的種子不互相推斷。未封存的全部依賴、外部資料與任意全域 RNG 不在完全重現保證內；舊結果缺證據時保留限制。
- **契約**：[可重現性](../../ARCHITECTURE.md#可重現性與策略執行安全)。
- **驗證**：[執行亂數](../../tests/app/test_execution_randomness.py)、[來源 lineage](../../tests/app/test_run_lineage_manifest.py) 核對受控設定與保存證據，不代表跨平台數值完全一致。
