# 共通品質需求

本冊適用 UC-01 至 UC-05；整體範圍見 [SRS](../requirements.md)。

## NFR-SAFE-001 程式執行與部署邊界

- **需求**：MCP、CLI 與高階 Python API 從首次匯入起統一隔離生成程式，限制其程序、檔案、網路與資源使用；靜態政策、執行邊界及 HTTP 入口保護須明確區分。
- **驗收**：生成程式在有獨立期限的 Linux 容器中以非特權身分執行，沒有主機工作區／daemon socket 掛載或外部網路，並套用記憶體、CPU、程序數、暫存及傳輸限制。缺少必要環境明確失敗，不回退原生執行。macOS 控制端可保持原生。固定執行綁定來源與映像／資源政策；不把靜態檢查、操作同意或容器當成全面安全證明。HTTP 的 Host／Origin 及本文容量仍依既有規則檢查，允許清單不取代登入與存取控制。
- **契約**：[策略安全模型](../strategy_contract.md#security-model)、[連線與部署](../../README.md#2-啟動-mcp-server-並連接助手)。
- **驗證**：[真實容器邊界](../../tests/integration/test_strategy_isolation.py)、[生成程式協定](../../tests/integration/test_generated_runtime.py)、[資料傳輸](../../tests/adapters/test_frame_codec.py)、[HTTP 入口保護](../../tests/mcp/test_http_transport_security.py)、[真實 HTTP 本文容量與確認](../../tests/mcp/test_http_execution_confirmation.py) 驗證適用邊界；不證明生成程式的全面安全。

## NFR-OPS-001 有限資源與失敗可觀察性

- **需求**：可成長的清單及輸出使用分頁、明確容量或 artifact；受監督的驗證與準備具期限，背景工作可取消並查詢清理結果。
- **驗收**：超量明確失敗或提供續讀，不默默截斷成果；有界診斷尾端須標明截斷。成功、例外、取消與逾時均核對所屬程序及隔離單元停止後清理臨時資料，包含另開 session 的生成策略後代。Host controller 消失不取消隔離單元的獨立期限；daemon 故障或歸屬不明時保留控制紀錄並回報未確認清理，不授予成功證據。保留使用者成果；可信 host worker 的強殺限制另行揭露。
- **契約**：[程序控制與清理](../../ARCHITECTURE.md#程序控制與清理)、[策略驗證限制](../strategy_contract.md#security-model)、[執行準備](../execution_plans.md#execution-plans-and-confirmation-documents)、[工作控制與清理狀態](../run_artifacts.md#job-state-and-worker-identity)、[探索分頁](../run_artifacts.md#discovery-pages)。
- **驗證**：[容器失敗／期限與後代](../../tests/integration/test_strategy_isolation.py)、[生成策略跨 session 清理](../../tests/integration/test_generated_worker_cleanup.py)、[控制協定與故障清理](../../tests/adapters/test_container_runtime.py)、[preflight 臨時檔案](../../tests/integration/test_preflight_temporary_files.py)、[真實受監督 worker](../../tests/integration/test_managed_workers.py)、[分頁](../../tests/app/test_pagination.py) 檢查清理、狀態與容量。

## NFR-REPRO-001 有界的重現性承諾

- **需求**：執行保存有效設定、版本、資料指紋與適用的隨機種子，讓研究可追溯；結果查詢及報告不因當前預設值改變而重新計算。
- **驗收**：相同受控條件下的種子使用可驗證；策略執行與獨立 ML 訓練的種子不互相推斷。未封存的全部依賴、外部資料與任意全域 RNG 不在完全重現保證內；舊結果缺證據時保留限制。
- **契約**：[可重現性](../../ARCHITECTURE.md#可重現性與策略執行安全)。
- **驗證**：[執行亂數](../../tests/app/test_execution_randomness.py)、[來源 lineage](../../tests/app/test_run_lineage_manifest.py) 核對受控設定與保存證據，不代表跨平台數值完全一致。
