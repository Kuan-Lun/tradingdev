# 需求維護要求

本冊定義需求與契約的維護能力；整體範圍見 [SRS](../requirements.md)。

## NFR-MAINT-001 需求與契約維護

- **需求**：產品需求保留穩定身分、驗收條件與詳細契約／驗證入口；欄位、型別及預設值維持正式定義，避免多份手寫 schema。
- **驗收**：重複編號、缺少必要欄位、斷裂導航／引用及未登錄的需求文件使文件檢查失敗。SRS 異動納入候選語意審查；必要的明示契約原文缺失時拒絕審查，不以連結或工具正常結束宣稱需求正確。
- **契約**：[SRS 檢查工具](../development.md#srs-維護與檢核)。
- **驗證**：[SRS 結構檢查](../../tests/tooling/test_requirements.py)、[候選分流](../../tests/tooling/test_change_scope.py)、[審查證據](../../tests/tooling/test_review_plan.py) 驗證機械規則與固定版本原文；語意審查仍可能漏判。
