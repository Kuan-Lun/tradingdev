<!-- presentation-url: https://slides.klwang.tw/tradingdev/ -->

<!--
marp: true
theme: tradingdev
size: 16:9
paginate: true
lang: zh-Hant
title: TradingDev｜能學習策略開發經驗、自我演進的研究後端
description: 產品介紹、使用場景、分類比較與後續開發方向
-->

<!-- _class: cover -->

# 自動整合多人研究成果而演進的量化交易策略開發 MCP 服務

能學習策略開發經驗、自我演進的研究後端

**永豐 x CCNS** 合作專案

{{presentation-qr}}

<!--
_footer: |-
  GitHub: [TradingDev](https://github.com/Kuan-Lun/tradingdev)\
  簡報網址：{{presentation-link}}
-->

---

<!-- _class: compact -->

## 參與成員

- **王冠倫** 國立臺灣大學資訊工程學系 博士 (應屆)
  - 實驗室：計算理論暨金融計算實驗室 (呂育道 教授)
- **許竣鈞** 國立成功大學經濟學系 學士班
  - 雙主修：資訊工程學系
- **Morten Weng** University of Bergen 博士
  - 應屆畢業
- **黃如君** 永豐銀行專案經理

---

<!-- _class: compact -->

## 閱讀路線

| 章節 | 主題 | 內容 |
| --- | --- | --- |
| 01 | [開發動機與專案定位](index.html#4) | 工具比較、工程回饋與使用情境 |
| 02 | [策略研究](index.html#11) | 角色分工、運算安排與策略實驗 |
| 03 | [後續開發方向](index.html#49) | 可獨立指定的開發目標 |
| Appendix 01 | [工具語意](index.html#52) | 註解語意、產品限制 |
| Appendix 02 | [架構與功能分析](index.html#55) | 研究條件、行情與回覆契約、安全與限制 |
| Appendix 03 | [保存設計與實作案例](index.html#74) | 保存設計、模組責任與兩個實作案例 |

<!-- _footer: "點選章名跳轉。方向鍵換頁，P 開啟講者模式。" -->

---

<!-- header: "01　開發動機與專案定位" -->
<!-- _class: chapter -->
<!-- _header: "" -->

## 01　開發動機與專案定位

研究背景、必要能力與本專案的研發定位。

---

<!-- _class: compact -->

## 開發動機與必要能力

> **建立共用研究後端，支援成員開發各自的交易策略。**

- **實驗基礎**：在同一後端執行回測，保存可重現的設定與結果。
- **策略與模型**：支援成員建構、修改與驗證不同的量化方法。
- **策略分析與修正**：檢查策略問題，分析實驗結果，協助下一輪修正。
- **後端改進**：透過提示詞，讓 AI 代理依研究經驗提出並驗證後端改進。

<!-- _footer: "開發目標，尚未全部完成。" -->

---

<!-- _class: compact -->

## 現成工具與本專案的整合方向

| 現成工具 | 已提供的基礎 |
| --- | --- |
| VectorBT、Backtrader | 回測計算與可擴充的策略框架 |
| VectorBT PRO MCP | AI 程式研究與按需研究指引 |
| QuantConnect | 研究代理、診斷、專案記憶與共用程式庫 |

**本專案提供 AI 輔助策略開發與研究的共用後端，
並支援將研究中的共通檢查與元件整合回後端，供後續策略使用。**

<!-- _footer: "來源：[VectorBT](https://vectorbt.dev/)、[Backtrader](https://www.backtrader.com/docu/)、[VBT PRO MCP](https://vectorbt.pro/features/ai/mcp-server/)、[QC Agents](https://www.quantconnect.com/docs/v2/ai-assistance/agents)、[程式庫](https://www.quantconnect.com/docs/v2/local-platform/projects/shared-libraries) · 查核 2026-10-31" -->

---

<!-- _class: compact -->

## 為何需要自行開發 TradingDev

**自行維護 TradingDev，讓研究經驗能回饋下一次開發。**

- **案例與證據**：連結問題、修正依據與適用條件。
- **修改與驗證**：AI 代理提出檢查或元件，經跨策略測試與審查整合。
- **追蹤與評估**：保留改進版本，檢查後續研究是否減少重複錯誤。

這需要掌握**研究證據、檢查規則與版本更新流程**。

---

## 一次修正如何幫助其他研究

1. 成員 A 發現資料時間對齊錯誤，保留可重現案例。
2. 人員提示 AI 代理，提出共用檢查與測試。
3. 其他策略通過回歸驗證，審查後更新 TradingDev。
4. 成員 B 開發不同策略時，提早攔下同類錯誤。

*情境示意；減少重複錯誤的效果尚待實測。*

---

## 目前已建立的研究基礎

> TradingDev 透過 MCP，讓 AI 代理運用研究指引建立與修正策略，
> 依據檢查與實驗結果，提出可追問的分析。

- **策略實驗**：建立與修正策略，完成檢查與歷史回測。
- **ML 訊號**：提供已登錄模型的訓練、預測與評估。
- **證據保存**：連結策略版本、有效設定與實驗結果。
- **結果分析與研究筆記**：提供診斷提示、報告解讀與可續讀的研究筆記。

已建立 [軟體需求規格（SRS）](https://github.com/Kuan-Lun/tradingdev/blob/main/docs/requirements.md)，整理產品需求與驗收條件。

<!-- _footer: "現況依據：[TradingDev 使用說明](https://github.com/Kuan-Lun/tradingdev/blob/cd95d6091bbc2a46c6a58cdb326aa5df730519c4/README.md)與[架構](https://github.com/Kuan-Lun/tradingdev/blob/cd95d6091bbc2a46c6a58cdb326aa5df730519c4/ARCHITECTURE.md)" -->

---

<!-- _class: compact -->

## TradingDev 的使用情境

| 情境 | 如何使用 |
| --- | --- |
| 個人研究 | 沿用自己的 AI 代理，提出策略並追問實驗結果 |
| 教學 | 用相同題目與資料，比較規則、成本及驗證方法 |
| 研究團隊 | 沿用團隊資料與實驗格式，累積共用檢查 |
| 公司產品 | 將策略實驗接進既有 AI 網站，安排模型與收費 |

*使用情境示意；教學與團隊可先採獨立環境。
多人權限、策略隔離與集中資源管理仍待開發。*

---

<!-- header: "02　策略研究" -->
<!-- _class: chapter -->
<!-- _header: "" -->

## 02　策略研究

一個交易想法如何成為可檢查、可追問的實驗。

---

## TradingDev 提供的研究流程指引

1. **釐清構想**：確認交易想法、標的、期間與 K 棒頻率。
2. **開發與修復**：依策略契約撰寫，根據檢查結果修正。
3. **確認實驗**：閱讀設定，決定是否正式執行。
4. **解讀證據**：依回測與診斷比較修改差異，區分觀察、假說及限制。
5. **延續研究**：提出下一步，依需要保存研究筆記。

<!-- _footer: "來源：[執行確認流程](https://github.com/Kuan-Lun/tradingdev/blob/35d709d4ac9ed8c9d3a71a3e24cd98996bf3a1e6/README.md#完成第一次回測)" -->

---

<!-- _class: compact -->

## 對話與運算的部署

對話可留在筆電，回測交給另一台主機。

| 可自行安排的部分 | 整合者的選擇 |
| --- | --- |
| 對話與模型 | 沿用個人使用的 AI 代理，或由公司提供模型及互動流程 |
| 資料與運算 | 提供自己的行情與主機，安排快取政策 |
| 研究規則與輸出 | 維護策略契約、檢查與報告呈現方式 |

*跨設備連線須設定網路轉送與存取控制。集中排程及多人隔離仍待開發。*

---

## 使用者、AI 代理與 TradingDev

| 角色 | 負責的事 |
| --- | --- |
| 使用者 | 提出目標、釐清規則、確認實驗 |
| AI 代理 | 依研究指引理解需求、撰寫與修正策略、解讀結果 |
| TradingDev | 研究流程指引、策略契約與檢查、實驗計算與證據 |

TradingDev 透過 **MCP 協定**，提供研究指引與工具給 Codex 等 AI 代理。

可直接使用既有客戶端，也可整合進自己的應用。

---

## 一次工具呼叫的內部路徑（1/2）

<!-- _class: compact diagram-short -->

```mermaid
sequenceDiagram
    accTitle: 一次工具呼叫在 TradingDev 內部的路徑：代理到應用層
    participant A as AI 代理
    participant MCP as MCP 工具
    participant App as Application services
    A->>MCP: 呼叫工具（參數與契約）
    MCP->>App: 整理輸入、檢查型別
    App->>App: 協調工作流程
```

MCP 工具負責協定與資料轉換；工作流程與狀態轉移由 application 層服務協調。

---

## 一次工具呼叫的內部路徑（2/2）

<!-- _class: compact diagram-short -->

```mermaid
sequenceDiagram
    accTitle: 一次工具呼叫在 TradingDev 內部的路徑：領域層到回傳
    participant App as Application services
    participant Dom as Domain／Adapters
    participant WS as Workspace
    App->>Dom: 協調策略／資料／執行
    Dom->>WS: 讀寫 revision／快取／結果
    WS-->>App: 回傳結構化結果
```

**Workspace 是獨立於程式之外的外部存放空間**（策略修訂、行情快取、執行結果），與前面各層的程式內運算層類型不同；Domain／Adapters 讀寫它，再由 MCP 回傳給 AI 代理。

---

<!-- _class: compact -->

## 開發經驗的整合與驗證

由人員選定研究案例，啟動 research-feedback Skill，讓能存取 TradingDev 原始碼的開發 AI 代理整合與驗證後端改進。

1. **挑選共通問題**：比較案例與既有元件，確認可共用範圍、保留策略假設。
2. **定義契約並整合**：明確輸入、輸出與適用條件，提供共用檢查或元件的使用入口。
3. **驗證使用端**：固定資料與設定，檢查其他策略、正常情境及既有行為。
4. **審查與版本更新**：完成獨立審查，記錄行為變更與版本，再依流程更新服務。

*跨研究採用，以及減少重複錯誤與人工工時的成效，仍待驗證。*

---

## 依研究目的選擇模式

| 模式 | 主要回答的問題 |
| --- | --- |
| **signal：資金表現** | 給定一筆起始資金，資產、報酬與回撤如何變化？ |
| **volume：交易量與成本** | 按指定交易規模運作，累計成交量、成本與金額損益如何變化？ |

*volume 不設定起始本金，金額損益不能解讀為本金報酬率。*

---

<!-- _class: compact -->
<!-- _header: "02　策略研究　／　0050 均線策略案例" -->

## 0050 均線策略：實際研究流程

> 「我想用均線來買賣 0050，可以幫我看看嗎？」

**研究條件**：0050 日 K，2024/1/1 至 7/1。

**案例內容**：在同一段會話中釐清需求、確認方案、比較結果，再討論改善與整理分享。

**研究成果**：[研究筆記](evidence/0050-taiex/cases/01-ma-discussion/notebook/index.html)、[下載研究分享包](evidence/0050-taiex/cases/01-ma-discussion/research.zip)

**過程與證據**：[完整對話](evidence/0050-taiex/cases/01-ma-discussion/conversation/index.html)、[案例總覽](evidence/0050-taiex/index.html)、[下載完整案例證據包](evidence/0050-taiex/natural-case-evidence.zip)

*下載包完整解壓後可離線閱讀；研究筆記為摘要，逐字內容由完整對話查閱。*

<!-- _footer: "真實模型回覆與 MCP 紀錄；示範操作者由 AI 代理依委託擔任。Codex CLI · gpt-6.1-sol / medium · 產品版本 4c8a665。" -->

---

<!-- _class: case-dialogue case-transcript case-options case-with-input -->
<!-- _header: "02　策略研究　／　0050 均線策略案例" -->

## 實際對話：從模糊構想開始

<!-- dialogue-message: turn-1/0 -->

操作者輸入

<!-- dialogue-text-start -->

> 我想用均線來買賣 0050，可以幫我看看嗎？

<!-- dialogue-text-end -->

<!-- dialogue-message: turn-1/1 -->

gpt-6.1-sol｜該輪最終回覆

<!-- dialogue-text-start -->

> 可以。我建議先研究「均線判斷趨勢」的方式：
>
> - **買進並持有**：短期均線高於長期均線。
> - **賣出轉空手**：短期均線低於長期均線，不做空。
>
> 這種方法可能跟上較長的漲勢，但盤整時容易反覆買賣，成本也會影響成果；目前還不能判斷是否適合 0050。
>
> 先請你選兩件事：
>
> 1\. **回測期間**：想看哪段時間？例如 2020～2025 年。
> 2\. **K 線週期**：想用日線還是週線？若偏向波段操作，我建議先從日線開始。
>
> 確認後，我會提出少量均線組合與完整回測方案供你檢視。資料可使用 Yahoo Finance 的 `0050.TW`，實際歷史涵蓋範圍仍待確認。

<!-- dialogue-text-end -->

<!-- _footer: "CLI 實際對話原文；操作者由 AI 代理依委託擔任。[本輪完整對話與工具紀錄](evidence/0050-taiex/cases/01-ma-discussion/conversation/index.html#turn-1)" -->

---

<!-- _class: compact diagram-tight -->

## UC-01（1/2）：釐清、建立與驗證策略

```mermaid
sequenceDiagram
    accTitle: UC-01 從構想到回測（釐清與建立）
    participant U as 使用者
    participant A as AI 代理
    participant T as TradingDev
    U->>A: 策略想法（標的／期間／K 棒頻率）
    A->>T: get_strategy_contract(section=research/core)
    A->>T: list_data_sources
    A->>U: 追問必要研究選擇
    U->>A: 補齊標的、期間、成本等
    A->>T: save_strategy
    A->>T: validate_strategy
    A->>T: dry_run_strategy
    A->>T: promote_strategy
```

缺項先追問；每個 `save_strategy` 回傳 `revision_id`，驗證與執行都綁定它。

---

<!-- _class: case-dialogue case-transcript case-options case-with-input case-preview -->
<!-- _header: "02　策略研究　／　0050 均線策略案例" -->

## 實際對話：選定期間與日 K

<!-- dialogue-message: turn-2/0 -->

操作者輸入

<!-- dialogue-text-start -->

> 先看 2024 年 1 月 1 日到 7 月 1 日，用日 K。

<!-- dialogue-text-end -->

<!-- dialogue-message: turn-2/1 -->

gpt-6.1-sol｜該輪最終回覆（節錄；確認單由連結續讀）

<!-- dialogue-text-start -->

> 已備妥三組均線比較方案。下列 UTC 起訖時間對應台灣時間 **2024/1/1 至 7/1 全天**。請閱讀確認單，再回覆「同意」、拒絕或修改要求；TradingDev 要求確認方案後才執行歷史回測。
>
> [閱讀確認單](evidence/0050-taiex/cases/01-ma-discussion/confirmation.html)

<!-- dialogue-text-end -->

[![0050 均線策略的回測確認單](assets/0050-taiex-confirmation.jpg)](evidence/0050-taiex/cases/01-ma-discussion/confirmation.html)

[回測確認單｜開啟完整內容](evidence/0050-taiex/cases/01-ma-discussion/confirmation.html)

<!-- _footer: "CLI 實際對話原文；操作者由 AI 代理依委託擔任。[本輪完整對話與工具紀錄](evidence/0050-taiex/cases/01-ma-discussion/conversation/index.html#turn-2)" -->

---

<!-- _class: compact diagram-slim -->

## 研究生命週期以「確認」為分界

三個分期以**確認回測參數**為界；前與後是兩個工具截然不同的世界。

```mermaid
sequenceDiagram
    accTitle: 研究生命週期：確認前的準備 vs 回測完的查詢
    participant 前 as 確認回測參數前
    participant 界 as 確認／執行邊界
    participant 後 as 回測完
    前->>前: 釐清、建立、驗證、準備
    前->>界: prepare_* 產生確認單
    界->>界: 使用者明確同意才執行
    界->>後: submit 建立背景 job
    後->>後: 監控、查詢、報告、分享
```

分界之前只做合成檢查、不下載正式行情、不跑回測；分界之後才讀取真實結果。修改或拒絕則回到準備，需重新確認。

<!-- _footer: "來源：[SERVER_INSTRUCTIONS 步驟 4–6](https://github.com/Kuan-Lun/tradingdev/blob/main/src/tradingdev/mcp/prompts.py)" -->

---

<!-- _class: case-dialogue case-transcript case-options case-with-input -->
<!-- _header: "02　策略研究　／　0050 均線策略案例" -->

## 實際對話：閱讀後同意執行（1/2）

<!-- dialogue-message: turn-3/0 -->

操作者輸入

<!-- dialogue-text-start -->

> 我同意

<!-- dialogue-text-end -->

<!-- dialogue-message: turn-3/1 -->

gpt-6.1-sol｜該輪最終回覆

<!-- dialogue-text-start -->

> 回測完成。0050 在 **2024/1/1～7/1、日 K** 的結果如下，已含方案中的模擬手續費與滑價：
>
> | 均線組合 | 區間報酬 | 最大回撤 | 已平倉筆數 |
> | --- | ---: | ---: | ---: |
> | 5／20 日 | 28.02% | 8.16% | 1 |
> | 10／40 日 | 16.96% | 7.84% | 1 |
> | 20／60 日 | 16.50% | 7.84% | 0 |
>
> **5／20 日在這段期間報酬最高，但樣本太少，還不能認定長期最好。** 三組期末都仍持倉，報酬包含未實現損益；長均線暖機也占用了較多觀察期間。模擬成本尚未另計證交稅。
>
> [閱讀完整回測報告](evidence/0050-taiex/cases/01-ma-discussion/earlier-reports/dca6bf24d022a15c2e10f306db5c2119b85a688b0d38fdb935a0b14d255fe55c.html)

<!-- dialogue-text-end -->

<!-- _footer: "CLI 實際對話原文；操作者由 AI 代理依委託擔任。[本輪完整對話與工具紀錄](evidence/0050-taiex/cases/01-ma-discussion/conversation/index.html#turn-3)" -->

---

<!-- _class: case-dialogue case-transcript case-options case-preview case-preview-pair -->
<!-- _header: "02　策略研究　／　0050 均線策略案例" -->

## 實際對話：閱讀後同意執行（2/2）

<!-- dialogue-message: turn-3/1 -->

gpt-6.1-sol｜該輪最終回覆

<!-- dialogue-text-start -->

> 點選總覽表格或結果清單中的結果列，可進入該組參數的詳情；再以「返回」或「切換結果」查看其他組。已選取的詳情章節：設定與資料覆蓋、績效分析與保存指標、基準比較（Benchmark）、權益與回撤、委託、成交與帳戶核對、配對交易損益。所有內容保存在同一份離線 HTML。
>
> 比較工作：`7c67128548d24541b68ab39faf778075`；分析範圍：全部三組的 `full`。

<!-- dialogue-text-end -->

[![5／20 日均線的權益與回撤曲線](assets/0050-taiex-equity.jpg)](evidence/0050-taiex/cases/01-ma-discussion/report.html#result-0-0) [![5／20 日均線的 QuantStats 績效分析](assets/0050-taiex-quantstats.jpg)](evidence/0050-taiex/cases/01-ma-discussion/report.html#result-0-0)

[權益與回撤｜5／20 日均線](evidence/0050-taiex/cases/01-ma-discussion/report.html#result-0-0) [QuantStats 績效分析｜5／20 日均線](evidence/0050-taiex/cases/01-ma-discussion/report.html#result-0-0)

<!-- _footer: "CLI 實際對話原文；操作者由 AI 代理依委託擔任。[本輪完整對話與工具紀錄](evidence/0050-taiex/cases/01-ma-discussion/conversation/index.html#turn-3)" -->

---

## UC-01（2/2）：準備、確認、執行與查詢

<!-- _class: compact diagram-tight -->

```mermaid
sequenceDiagram
    accTitle: UC-01 從構想到回測（準備到查詢）
    participant U as 使用者
    participant A as AI 代理
    participant T as TradingDev
    A->>T: save_research_evidence
    A->>T: prepare_backtest
    T->>A: 確認單（confirmation_artifact）＋摘要
    A->>U: 顯示計畫，等待決定
    U->>A: 明確同意
    A->>T: submit_execution_decision(approve)
    T->>A: 建立背景 job
    A->>T: get_job_status（輪詢）
    T->>A: done 與 run_id
    A->>T: get_run_metrics／generate_report
    A->>U: 解讀結果與報告連結
```

準備本身不執行回測；只有使用者明確同意後才提交並建立背景工作。

---

<!-- _class: compact diagram-mid -->

## UC-03：找回歷史研究（不重跑）

```mermaid
sequenceDiagram
    accTitle: UC-03 找回歷史研究
    participant U as 使用者
    participant A as AI 代理
    participant T as TradingDev
    U->>A: 找回之前的研究結果
    A->>T: list_runs／get_run
    A->>T: get_run_metrics（含 scope）
    A->>T: get_run_trades／equity／executions
    A->>T: generate_report（run_ids）
    T->>A: 離線報告 artifact
    A->>U: 解讀與報告連結
```

以保存內容產生報告；缺少或損壞證據會明示，不悄悄重建歷史。

---

<!-- _class: case-dialogue case-transcript case-options case-with-input -->
<!-- _header: "02　策略研究　／　0050 均線策略案例" -->

## 實際對話：詢問台股大盤比較（1/2）

<!-- dialogue-message: turn-4/0 -->

操作者輸入

<!-- dialogue-text-start -->

> 這三組策略跟台股大盤相比，表現如何？

<!-- dialogue-text-end -->

<!-- dialogue-message: turn-4/1 -->

gpt-6.1-sol｜該輪最終回覆

<!-- dialogue-text-start -->

> **這段期間三組都落後台股大盤，最大回撤也較大。** 以台股加權指數（`^TWII`）為基準，實際比較期間為 2024/1/2～7/1，共 118 根日 K。
>
> | 策略／基準 | 區間報酬 | 與大盤差距 | 最大回撤 |
> | --- | ---: | ---: | ---: |
> | 台股大盤基準 | 29.01% | — | 6.66% |
> | 5／20 日均線 | 28.02% | −0.99 個百分點 | 8.16% |
> | 10／40 日均線 | 16.96% | −12.05 個百分點 | 7.84% |
> | 20／60 日均線 | 16.50% | −12.51 個百分點 | 7.84% |
>
> 5／20 日最接近大盤，但本次沒有展現較低回撤的優勢。另兩組暖機較長、較晚開始交易，也會影響比較結果，不能直接推論長期表現。

<!-- dialogue-text-end -->

<!-- _footer: "CLI 實際對話原文；操作者由 AI 代理依委託擔任。[本輪完整對話與工具紀錄](evidence/0050-taiex/cases/01-ma-discussion/conversation/index.html#turn-4)" -->

---

<!-- _class: case-dialogue case-transcript case-options case-preview case-preview-compact -->
<!-- _header: "02　策略研究　／　0050 均線策略案例" -->

## 實際對話：詢問台股大盤比較（2/2）

<!-- dialogue-message: turn-4/1 -->

gpt-6.1-sol｜該輪最終回覆

<!-- dialogue-text-start -->

> 大盤基準採首根收盤買進、期末估值，套用相同模擬費率與滑價；使用的是**不含股利的價格指數**。要進一步判斷均線擇時有沒有價值，還應比較「0050 本身買入持有」。
>
> [閱讀完整比較報告](evidence/0050-taiex/cases/01-ma-discussion/report.html)
>
> 點選總覽表格或結果清單中的結果列，可進入該組參數的詳情；再以「返回」或「切換結果」查看其他組。已選取的詳情章節：設定與資料覆蓋、績效分析與保存指標、基準比較（Benchmark）、權益與回撤、委託、成交與帳戶核對、配對交易損益。所有內容保存在同一份離線 HTML。
>
> 比較工作：`7c67128548d24541b68ab39faf778075`；範圍：三組 `full`。

<!-- dialogue-text-end -->

[![5／20 日均線與台股加權指數的基準比較表](assets/0050-taiex-benchmark.jpg)](evidence/0050-taiex/cases/01-ma-discussion/report.html#result-0-0)

[台股大盤比較｜5／20 日均線](evidence/0050-taiex/cases/01-ma-discussion/report.html#result-0-0) · [基準資料](evidence/0050-taiex/cases/01-ma-discussion/benchmark.json)

<!-- _footer: "CLI 實際對話原文；操作者由 AI 代理依委託擔任。[本輪完整對話與工具紀錄](evidence/0050-taiex/cases/01-ma-discussion/conversation/index.html#turn-4)" -->

---

<!-- _class: compact diagram-dense -->

## UC-02：比較、Walk-forward 與最佳化

```mermaid
sequenceDiagram
    accTitle: UC-02 比較與樣本外研究
    participant U as 使用者
    participant A as AI 代理
    participant T as TradingDev
    U->>A: 比較多組／做樣本外／搜尋參數
    A->>T: save_research_evidence
    A->>T: prepare_comparison 或 walk_forward 或 optimization
    T->>A: 確認單（含模式與期間）
    A->>U: 顯示計畫
    U->>A: 同意
    A->>T: submit_execution_decision(approve)
    T->>A: 建立 job（比較含父子候選）
    A->>T: get_job_status
    T->>A: done
    A->>T: generate_report(comparison_job_id=job_id)
    A->>U: 比較報告與排名
```

三種 `prepare_*` 回答不同問題；最佳化結果不直接等同樣本外保證。

---

<!-- _class: case-dialogue case-transcript case-options case-with-input -->
<!-- _header: "02　策略研究　／　0050 均線策略案例" -->

## 實際對話：討論改善方向（1/2）

<!-- dialogue-message: turn-5/0 -->

操作者輸入

<!-- dialogue-text-start -->

> 如果要改善這個策略，你會怎麼建議？

<!-- dialogue-text-end -->

<!-- dialogue-message: turn-5/1 -->

gpt-6.1-sol｜該輪最終回覆

<!-- dialogue-text-start -->

> 我會優先改善「驗證方式與風險控制」，目前還不急著調出另一組均線。這次只有半年資料、各組僅 0～1 筆已平倉交易，證據還太少。
>
> 建議依序做：
>
> 1\. **先比較 0050 買入持有**
> 目前比較的是加權指數。加入 0050 本身的基準，才能看出均線進出的貢獻。
>
> 2\. **統一暖機，再拉長測試期間**
> 在評估開始前提供足夠資料計算均線，讓三組從同一天開始比較；再涵蓋上漲、下跌與盤整期間，保留一段資料做樣本外驗證。

<!-- dialogue-text-end -->

<!-- _footer: "CLI 實際對話原文；操作者由 AI 代理依委託擔任。[本輪完整對話與工具紀錄](evidence/0050-taiex/cases/01-ma-discussion/conversation/index.html#turn-5)" -->

---

<!-- _class: case-dialogue case-transcript case-options -->
<!-- _header: "02　策略研究　／　0050 均線策略案例" -->

## 實際對話：討論改善方向（2/2）

<!-- dialogue-message: turn-5/1 -->

gpt-6.1-sol｜該輪最終回覆

<!-- dialogue-text-start -->

> 3\. **一次只測一種規則改善**
> 我會先測「短均線連續兩天低於長均線才賣出」，觀察能否減少短暫跌破造成的反覆交易。代價是出場較慢，可能承受更大跌幅，效果仍待驗證。
>
> 4\. **補齊實際交易成本**
> 納入證交稅、券商最低手續費與整股限制，確認改善後的收益是否留得住。
>
> **我會先保留 5／20 日作為研究起點，再與加入兩天出場確認的版本比較。** 它只是本次最接近大盤的組合，尚未證明長期較好；評估時應同時看報酬、回撤與交易次數。以上尚未執行新回測。

<!-- dialogue-text-end -->

<!-- _footer: "CLI 實際對話原文；操作者由 AI 代理依委託擔任。[本輪完整對話與工具紀錄](evidence/0050-taiex/cases/01-ma-discussion/conversation/index.html#turn-5)" -->

---

<!-- _class: case-dialogue case-transcript case-options case-with-input -->
<!-- _header: "02　策略研究　／　0050 均線策略案例" -->

## 實際對話：整理研究與分享

<!-- dialogue-message: turn-6/0 -->

操作者輸入

<!-- dialogue-text-start -->

> 幫我把這次的研究整理好，讓我能分享給別人。

<!-- dialogue-text-end -->

<!-- dialogue-message: turn-6/1 -->

gpt-6.1-sol｜該輪最終回覆

<!-- dialogue-text-start -->

> 已整理成**第 1 版研究分享包**，包含研究設定、三組回測、大盤比較、完整報告，以及尚未驗證的改善建議。
>
> [下載研究分享包 ZIP](evidence/0050-taiex/cases/01-ma-discussion/research.zip)
>
> 將整份 ZIP 分享給對方，**完整解壓後開啟 `index.html`**；保留 `reports/` 資料夾即可離線閱讀完整報告。
>
> 摘要已清楚標示主要結論：這段期間三組均落後大盤，且交易樣本少，不能據此認定長期優劣。

<!-- dialogue-text-end -->

<!-- _footer: "CLI 實際對話原文；操作者由 AI 代理依委託擔任。[本輪完整對話與工具紀錄](evidence/0050-taiex/cases/01-ma-discussion/conversation/index.html#turn-6)" -->

---

<!-- _class: compact diagram-mid -->

## UC-05：保存與分享研究

```mermaid
sequenceDiagram
    accTitle: UC-05 保存與分享
    participant U as 使用者
    participant A as AI 代理
    participant T as TradingDev
    U->>A: 整理並分享這次研究
    A->>T: save_research_note
    A->>T: export_research_note
    T->>A: 研究分享包 ZIP artifact
    A->>T: get_artifact_download（HTTP 連結）
    A->>U: 解釋解壓與開啟 index.html
```

研究筆記為摘要，非逐字紀錄；分享包完整解壓後離線可讀。

---

<!-- _class: compact -->

## 五個研究目的，對應不同工具群

- **把構想變成可執行策略**：`strategy` 群（建立、修訂、驗證、提升）。
- **把策略跑成可信實驗**：`execution` 與 `jobs` 群（準備、確認、執行、監控）。
- **把結果看清楚並能比較**：`runs`、`history`、`artifacts` 群（查詢、分析、報告、交付）。
- **把行情與特徵準備好**：`data` 與 `ml` 群（快取、獨立訊號研究）。
- **把研究留下來並能分享**：`research_notes`、`feature_requests`、`strategy_cleanup` 群。

**代理在每個目的下都先看再寫：先用唯讀工具確認，再用寫入工具推進。**

<!-- _footer: "來源：[SERVER_INSTRUCTIONS 九步流程](https://github.com/Kuan-Lun/tradingdev/blob/main/src/tradingdev/mcp/prompts.py)" -->

---

<!-- _class: compact -->

## 依研究目的劃分的工具群（1/2）

| 工具群 | 研究目的 | 工具數 | 讀寫 |
| --- | --- | ---: | --- |
| Strategy | 把構想寫成可檢查、可執行的策略 | 7 | 3 讀 / 4 寫 |
| Execution | 固定研究條件並產生可確認的計畫 | 8 | 1 讀 / 7 寫 |
| Data | 取得或確認行情與特徵資料 | 4 | 3 讀 / 1 寫 |
| Jobs | 追蹤與管理背景工作 | 3 | 1 讀 / 2 寫 |
| Runs | 查詢已保存的回測結果與指標 | 5 | 5 讀 |
| History | 讀取交易紀錄、診斷與組裝報告 | 8 | 7 讀 / 1 寫 |

<!-- _footer: "數量依 @mcp.tool 裝飾函式統計；讀寫以 ToolAnnotations.read_only_hint 為準。" -->

---

<!-- _class: compact -->

## 依研究目的劃分的工具群（2/2）

| 工具群 | 研究目的 | 工具數 | 讀寫 |
| --- | --- | ---: | --- |
| Artifacts | 列出與交付已保存產物 | 3 | 3 讀 |
| Research Notes | 留下可續讀的研究筆記 | 4 | 2 讀 / 2 寫 |
| Feature Requests | 記錄暫不支援的能力 | 2 | 1 讀 / 1 寫 |
| Strategy Cleanup | 預覽並清理未用草稿 | 1 | 1 寫 |
| ML | 獨立訓練與評估訊號 | 7 | 3 讀 / 4 寫 |
| **合計** | — | **52** | **29 讀 / 23 寫** |

<!-- _footer: "數量依 @mcp.tool 裝飾函式統計；讀寫以 ToolAnnotations.read_only_hint 為準。" -->

---

<!-- _class: statement -->

## 目前共 52 個 MCP 工具

> **29 個唯讀工具、23 個會建立或變更狀態的工具。**

工具依責任分成 11 群；同一群內同時有讀取與寫入，方便代理在對話中反覆查詢與遞進修改。

---

<!-- _class: compact -->

## 這 52 個工具分在哪裡（1/2）

| 模組 | 工具數 | 主要責任 |
| --- | ---: | --- |
| `strategy` | 7 | 建立、修訂、查詢與驗證策略 |
| `execution` | 8 | 準備回測、比較、Walk-forward、最佳化與確認 |
| `data` | 4 | 探索、查詢與快取行情 |
| `jobs` | 3 | 查詢與管理背景工作 |
| `runs` | 5 | 查詢回測結果、指標與比較 |
| `history` | 8 | 交易紀錄、診斷與報告 |

<!-- _footer: "來源：https://github.com/Kuan-Lun/tradingdev/blob/main/src/tradingdev/mcp/tools/" -->

---

<!-- _class: compact -->

## 這 52 個工具分在哪裡（2/2）

| 模組 | 工具數 | 主要責任 |
| --- | ---: | --- |
| `artifacts` | 3 | 列出與交付已保存產物 |
| `research_notes` | 4 | 保存與匯出研究筆記 |
| `feature_requests` | 2 | 記錄未支援能力 |
| `strategy_cleanup` | 1 | 預覽與清理草稿 |
| `ml` | 7 | 獨立資料集、模型、訊號評估與排名 |

前 6 群負責「把構想與研究條件變成可執行物件」；後 5 群負責「把結果留下並可分享」。共 11 群、52 個工具。

<!-- _footer: "來源：https://github.com/Kuan-Lun/tradingdev/blob/main/src/tradingdev/mcp/tools/" -->

---

<!-- _class: compact -->

## 五個使用案例對應的工具流程（1/2）

| 案例 | 觸發的階段 | 關鍵寫入工具 |
| --- | --- | --- |
| UC-01 從構想到回測 | 釐清 → 建立 → 驗證 → 準備 → 確認 → 執行 → 查詢 | `save_strategy`、`prepare_backtest`、`submit_execution_decision` |
| UC-02 比較與樣本外研究 | 準備比較／Walk-forward／最佳化 → 確認 → 執行 | `prepare_comparison` / `walk_forward` / `optimization` |
| UC-03 找回歷史研究 | 查詢 → 報告（不重跑） | `generate_report`（run_ids） |

<!-- _footer: "來源：https://github.com/Kuan-Lun/tradingdev/blob/main/docs/requirements.md#5-使用案例" -->

---

<!-- _class: compact -->

## 五個使用案例對應的工具流程（2/2）

| 案例 | 觸發的階段 | 關鍵寫入工具 |
| --- | --- | --- |
| UC-04 獨立 ML 訊號研究 | 合約 → 資料 → 資料集 → 訓練 → 評估 → 排名 | `build_dataset`、`train_model`、`evaluate_signal` |
| UC-05 保存與分享 | 筆記 → 匯出 → 交付 | `save_research_note`、`export_research_note` |

五個案例都遵循「先唯讀確認、再明確同意後寫入」；UC-03 不重跑回測，UC-04 走獨立 ML 軌道。

<!-- _footer: "來源：https://github.com/Kuan-Lun/tradingdev/blob/main/docs/requirements.md#5-使用案例" -->

---

<!-- _class: compact stage-tbl -->

## 分期與階段：執行前（1/2）

| 分期 | 研究階段 | 主要工具 |
| --- | --- | --- |
| 執行前 | 需求釐清 | `get_strategy_contract`、`list_data_sources`、`list_strategies`、`get_strategy` |
| 執行前 | 策略建立與驗證 | `save_strategy`、`validate_strategy`、`dry_run_strategy`、`promote_strategy` |

取得確認回測參數前，代理交錯呼叫多個階段；**只做合成檢查，不下載正式行情、不跑回測**。

`get_strategy_contract`、`list_data_sources`、`get_artifact`(_download) 為跨鄰近階段重複呼叫的參考與交付工具；其餘工具依主要階段歸屬，各階段不會完全重疊。

<!-- _footer: "來源：[requirements.md 使用案例](https://github.com/Kuan-Lun/tradingdev/blob/main/docs/requirements.md#5-使用案例) · [SERVER_INSTRUCTIONS](https://github.com/Kuan-Lun/tradingdev/blob/main/src/tradingdev/mcp/prompts.py)" -->

---

<!-- _class: compact stage-tbl -->

## 分期與階段：執行前（2/2）

| 分期 | 研究階段 | 主要工具 |
| --- | --- | --- |
| 執行前 | 資料準備 | `list_available_data`、`inspect_dataset`、`ensure_data` |
| 執行前 | 執行準備與確認 | `save_research_evidence`、`prepare_backtest` / `comparison` / `walk_forward` / `optimization`、`get_execution_plan`、`submit_execution_decision`、`cancel_execution_plan` |

這一階段以 `prepare_*` 產生確認單，等待使用者明確同意；**準備本身不執行回測**。

<!-- _footer: "來源：[requirements.md 使用案例](https://github.com/Kuan-Lun/tradingdev/blob/main/docs/requirements.md#5-使用案例) · [SERVER_INSTRUCTIONS](https://github.com/Kuan-Lun/tradingdev/blob/main/src/tradingdev/mcp/prompts.py)" -->

---

<!-- _class: compact stage-tbl -->

## 分期與階段：執行中／後（1/2）

| 分期 | 研究階段 | 主要工具 |
| --- | --- | --- |
| 執行中 | 執行監控 | `get_job_status`、`list_jobs`、`cancel_job` |
| 執行後 | 結果查詢與分析 | `list_runs`、`get_run`、`get_run_metrics`、`get_metric_catalog`、`compare_runs`、`find_runs`、`get_run_trades` / `equity` / `executions` / `account_history`、`get_run_diagnostics` |

確認後才讀取真實結果；此分期的工具多為唯讀，與執行前以寫入為主明顯不同。

<!-- _footer: "來源：[requirements.md 使用案例](https://github.com/Kuan-Lun/tradingdev/blob/main/docs/requirements.md#5-使用案例) · [SERVER_INSTRUCTIONS](https://github.com/Kuan-Lun/tradingdev/blob/main/src/tradingdev/mcp/prompts.py)" -->

---

<!-- _class: compact stage-tbl -->

## 分期與階段：執行後（2/2）與獨立軌道

| 分期 | 研究階段 | 主要工具 |
| --- | --- | --- |
| 執行後 | 報告與交付 | `get_report_sections`、`generate_report`、`list_artifacts`、`get_artifact`(_download) |
| 執行後 | 保存與分享 | `save_research_note`、`get_research_notes`、`export_research_note`、`get_research_archive`、`record_feature_request`、`list_feature_requests`、`cleanup_strategy_drafts` |
| 獨立軌道 | ML 訊號研究 | `get_ml_contract`、`build_dataset` / `build_panel`、`train_model`、`predict`、`evaluate_signal`、`leaderboard` |

<!-- _footer: "來源：[requirements.md 使用案例](https://github.com/Kuan-Lun/tradingdev/blob/main/docs/requirements.md#5-使用案例) · [SERVER_INSTRUCTIONS](https://github.com/Kuan-Lun/tradingdev/blob/main/src/tradingdev/mcp/prompts.py)" -->

---

<!-- _class: compact -->

## 確認前：代理在執行前交錯呼叫多群工具

以 UC-01「從構想到回測」為例，代理在**確認回測參數之前**依序穿過多個階段：

1. **釐清**：`get_strategy_contract`、`list_data_sources` 確認選擇與能力。
2. **建立與驗證**：`save_strategy` → `validate_strategy` → `dry_run_strategy` → `promote_strategy`。
3. **準備與確認**：`save_research_evidence` → `prepare_backtest` → `get_execution_plan`。
4. **執行與監控**：`submit_execution_decision` → `get_job_status`。
5. **查詢與報告**：`get_run_metrics` → `generate_report` → `get_artifact_download`。

**寫入工具一律在前一步的唯讀確認之後才呼叫。**

<!-- _footer: "來源：[UC-01 正常流程](https://github.com/Kuan-Lun/tradingdev/blob/main/docs/requirements.md#5-使用案例)" -->

---

<!-- _class: compact diagram-mid -->

## UC-04：獨立 ML 訊號研究

```mermaid
sequenceDiagram
    accTitle: UC-04 獨立 ML 訊號研究
    participant U as 使用者
    participant A as AI 代理
    participant T as TradingDev
    U->>A: 想做獨立訊號研究
    A->>T: get_ml_contract
    A->>T: ensure_data（明確下載）
    A->>T: build_dataset／build_panel
    A->>T: train_model
    A->>T: predict
    A->>T: evaluate_signal(record=true)
    A->>T: leaderboard
    A->>U: 評估與排名
```

ML 研究是同步獨立介面，不會自動轉成交易策略或建立一般回測 run。

---

<!-- header: "03　後續開發方向" -->
<!-- _class: chapter -->
<!-- _header: "" -->

## 03　後續開發方向

以現有成果為基礎，安排下一階段的開發與驗證。

---

<!-- _class: compact -->

## 目前提供與後續擴充

| 已提供 | 後續方向 |
| --- | --- |
| 固定參數批次執行、歷史結果比較與排序 | 依跨實驗證據選擇下一個實驗 |
| 規則式診斷、AI 解讀與研究建議 | 市場情境與波動敏感度診斷 |
| 研究筆記、報告引用、續讀與匯出 | 跨研究自動檢索與假說追蹤 |
| 各次回測獨立記錄設定、成本與結果 | 多策略組合與資金配置 |
| 工程回饋 Skill 與 TradingDev 整合基礎 | 跨研究採用與成效驗證 |
| HTTP／Docker 部署、策略及實驗編號 | 多人權限與策略隔離 |

*後續項目尚未完成，開發時程未定。*

<!-- _footer: "現況查核：2026-11-05／[TradingDev 7e7a997](https://github.com/Kuan-Lun/tradingdev/tree/7e7a997c67a6e9765b124c80286ddc75f8b506d7)" -->

---

<!-- header: "03　後續開發方向" -->

## 依使用場合決定優先序

| 使用場合 | 優先處理 |
| --- | --- |
| 個人反覆研究 | 進階診斷、實驗引導與跨研究檢索 |
| 團隊共享開發經驗 | 共用元件採用與工程回饋成效驗證 |
| 多個策略一起評估 | 組合分析與配置模型 |
| 對外提供多人服務 | 身分、權限與執行隔離 |

對話介面、模型路由與計費可留在整合層。
真實下單維持目前產品邊界之外。

---

<!-- header: "Appendix 01　工具語意" -->
<!-- _class: chapter -->
<!-- _header: "" -->

## Appendix 01　工具語意

AI 代理能叫到哪些「種類」的工具，以及工具不能保證什麼。

---

<!-- _class: compact -->

## 工具註解告訴代理「這是哪一類工具」

每個工具帶有四個 hint，協助代理決定呼叫時機與風險：

- **read_only_hint**：只讀取、不變更伺服器狀態（`list_*`、`get_*`）。
- **destructive_hint**：可能建立、覆寫或刪除（`save_strategy`、`ensure_data`、`cleanup_strategy_drafts`）。
- **idempotent_hint**：重複呼叫結果一致（`get_*`、`promote_strategy`、`generate_report`）。
- **open_world_hint**：可能執行生成程式或存取外部資料（`validate_strategy`、`ensure_data`）。

**hint 是對代理的提示，不代表後端已證明行為；權限仍由後端強制。**

<!-- _footer: "來源：https://github.com/Kuan-Lun/tradingdev/blob/main/src/tradingdev/mcp/tools/" -->

---

<!-- _class: compact -->

## 工具能做的與不能保證的

- TradingDev 研究歷史資料，**不提供真實下單**。
- 生成程式的檢查在 Linux 容器中執行，**不是完整安全 sandbox**。
- `prepare_*` 只做合成檢查，**不預測真實績效、不下載正式行情**。
- 確認單的「初始請求」不代表同意；**只有明確同意才提交執行**。
- 指標受資料頻率、成本假設限制；**缺失值不是零，也不重建歷史**。

---

<!-- header: "Appendix 02　架構與功能分析" -->
<!-- _class: chapter -->
<!-- _header: "" -->

## Appendix 02　架構與功能分析

研究條件、系統設計與實作案例

<!-- _footer: "分析基準：TradingDev f328b8f（2026-11-09）" -->

---

<!-- _class: compact -->

## 四個選擇由使用者決定，其餘由政策補齊並凍結（1/2）

**ResearchSpec 必填**：① 策略構想　② 標的　③ 期間（起訖兩欄）　④ K 棒頻率
缺項要追問，不從範例或可用資料推測；資料來源與市場類型也要明確選定。

> **優先序：** 明示值（含 0 成本）→ 明確選定的 revision → 共用政策；解析結果連同來源凍結，worker 不重套新政策。

<!-- _footer: "來源：docs/strategy_contract.md 的 Research Inputs and Simulation Policy、domain/research_policy.py" -->

---

<!-- _class: compact -->

## 四個選擇由使用者決定，其餘由政策補齊並凍結（2/2）

**未填時的共用政策（research-v1）**，這些是模擬假設，不是交易所現行費率：

| 項目 | 預設 |
| --- | --- |
| 本金 | 10,000（報價幣別，signal 模式） |
| 手續費率 | 0.0006 |
| 滑價率 | 0.0005 |
| 年化 | 只有純加密貨幣來源才推定 365；Yahoo 等需明示 |

<!-- _footer: "來源：docs/strategy_contract.md 的 Research Inputs and Simulation Policy、domain/research_policy.py" -->

---

<!-- _class: compact -->

## 引擎的停損另有規則：跳空越過門檻就以開盤價成交（1/2）

| 規則 | signal 模式 | volume 模式 |
| --- | --- | --- |
| 參考價 | 進場那根的收盤價 | 進場價 |
| 檢查範圍 | 之後每根的開盤到收盤，不看最高、最低 | 當根的最高價與最低價 |
| 成交價 | 跳空越過門檻用開盤價，否則用門檻價 | 門檻價 |
| 同時觸發 | 停損優先於停利與一般訊號 | 停損優先於停利 |
| 設為 0 | 零距離門檻，不是關閉 | 關閉 |

<!-- _footer: "來源：docs/strategy_contract.md 的 Signal and Execution Semantics 與 volume 停損、熔斷說明" -->

---

<!-- _class: compact -->

## 引擎的停損另有規則：跳空越過門檻就以開盤價成交（2/2）

| 規則 | signal 模式 | volume 模式 |
| --- | --- | --- |
| 停損後再進場 | 要等訊號狀態改變 | re_entry_after_sl 決定是否當根收盤再進場 |
| 成本 | 停損單不計出場滑價，仍計手續費 | 每次進出都計手續費與滑價 |

- **volume 的月虧損熔斷**：monthly_max_loss：每個 UTC 月的累計損益達門檻，就在收盤平倉並暫停到下個月；出場成本可能讓虧損超過門檻
- **兩種模式共通**：進場那根不檢查停損；都沒有完整的保證金、強平或資金費率模型；是歷史模擬假設，不是交易所成交保證

<!-- _footer: "來源：docs/strategy_contract.md 的 Signal and Execution Semantics 與 volume 停損、熔斷說明" -->

---

<!-- _class: compact -->

## 同一個資料身分，載入、取樣與檢查共用

| Binance Vision | Binance API | Yahoo Finance | 特徵資料 |
|---|---|---|---|
| 加密貨幣，歷史壓縮檔 | 加密貨幣 | 股票、ETF、期貨、指數、外匯 | DVOL 取自 Deribit；資金費率等用本機 Parquet |

**依 UTC 年度查詢** → **鎖住該身分與年度** → **寫入不可變版本** → **原子替換年度憑證**

- **資料身分**：來源＋市場類型＋原始代號＋頻率；不同來源不共用價格
- **重用條件**：查詢時間晚於區間結束 7 天以上才重用；近期區間每次重抓
- **時間對齊**：只沿用較早的有效特徵；開頭缺值保持未知，不回填
- **受限取樣**：確認前的取樣在請求階段就限量；下載、解壓與列數各有上限

<!-- _footer: "來源：ARCHITECTURE.md「行情快取」「獨立行情取樣」、docs/run_artifacts.md 的 Market Data Cache" -->

---

<!-- _class: compact -->

## 回覆有固定形狀，失敗分成兩種（1/2）

**形狀：**

- StrictMCPServer 拒絕未知的頂層參數
- 客戶端依 tools/list 的 outputSchema 讀 structuredContent
- 單一物件放根層；清單與成功／失敗聯集放在 result
- 可成長的清單分頁回摘要，細節另外查

<!-- _footer: "來源：ARCHITECTURE.md「MCP 回覆契約」；分頁欄位見 docs/run_artifacts.md 的 Discovery Pages" -->

---

<!-- _class: compact -->

## 回覆有固定形狀，失敗分成兩種（2/2）

**失敗：**

- **預期的應用失敗**：回傳穩定的 code 與 diagnostics，例如 strategy_requires_training
- **MCP error**：參數錯誤、未預期例外、無效回覆；內部原因只留在 server 日誌
- 只看 isError 不能判斷應用是否成功

> **標註不是授權：** 工具的唯讀、破壞性、冪等標註只描述副作用。狀態查詢可能更新失聯的 job，驗證會執行 Python。

<!-- _footer: "來源：ARCHITECTURE.md「MCP 回覆契約」；分頁欄位見 docs/run_artifacts.md 的 Discovery Pages" -->

---

<!-- _class: compact -->

## ML 訓練不偷看答案，排名只比能比的（1/2）

**ML 訓練：多檔一起學：**

- 幾檔股票排成同一張表一起訓練
- 驗證時，碰到測試期答案的資料一律剔除
- 日期對不齊時，不拿未來的資料補洞
- 驗證分數只供參考，最後保存的模型用全部資料重訓

<!-- _footer: "來源：docs/ml_research.md、docs/run_artifacts.md（main f328b8f）" -->

---

<!-- _class: compact -->

## ML 訓練不偷看答案，排名只比能比的（2/2）

**排名：只比能比的：**

- ML 訊號和回測結果用同一套規則排名
- 算不出的數字不當 0，條件不同不混排
- 越大越好或越小越好事先定好，同分同名次
- 只有一個候選時會註明：單一筆不能證明比較好

> **要記得：** ML 評估是不含交易成本的簡化統計，排名靠前不代表已經是能交易的策略。

<!-- _footer: "來源：docs/ml_research.md、docs/run_artifacts.md（main f328b8f）" -->

---

<!-- _class: compact -->

## 六條固定門檻，只提示該查什麼，不下結論（1/2）

| 規則 | 觸發條件 | 建議的下一步 |
| --- | --- | --- |
| `drawdown` | 最大回撤 > 25% | 查回撤期間的交易，先和原版比較一種明確的出場或風控變體 |
| `win_rate` | 勝率 < 45% | 先看虧損的進場與盈虧大小；勝率本身不代表能否獲利 |
| `few_trades` | 交易數 < 20 | 檢查資料涵蓋與訊號條件，再用更大的授權樣本或敏感度實驗 |
| `many_trades` | 交易數 > 200 | 先看周轉與成本；交易次數多不等於過度交易 |
| `profit_factor` | 獲利因子 < 1.2 | 看虧損大小與出場，在相同成本與期間下比較一種出場變體 |
| `sharpe` | Sharpe < 1.0 | 先看報酬波動與曝險；這條規則沒有量測波動敏感度 |

<!-- _footer: "來源：adapters/reporting/maverick_diagnostics.py；由 get_run_diagnostics 查詢" -->

---

<!-- _class: compact -->

## 六條固定門檻，只提示該查什麼，不下結論（2/2）

- **不評估的情況**：volume 模式不評估回撤與 Sharpe；彙總 scope 要回到原始結果；指標缺值標為不可用；風險報酬比與 Kelly 不在保存的指標內
- **回覆自帶的限制**：固定門檻不是使用者的風險上限；次數不依樣本長度調整；沒有觸發不代表策略安全；建議的實驗未經測試，也不授權執行

<!-- _footer: "來源：adapters/reporting/maverick_diagnostics.py；由 get_run_diagnostics 查詢" -->

---

<!-- _class: compact -->

## 結果、筆記與對話都能離線閱讀與分享

- **HTML 報告**：ReportService 供 MCP、report CLI、Dashboard 共用；總覽連到各結果明細，固定比較保留失敗原因；發佈後不改寫
- **研究筆記與封存**：LLM 寫草稿，後端以版本號防覆寫；封存 ZIP 核對附件路徑與 SHA-256
- **對話紀錄**：不綁供應商的回合模型；tradingdev-codex 驅動 Codex app-server 並保存可見事件，不含私有推理

HTTP 模式用短效簽章網址下載，簽發與下載都核對檔案；stdio 模式回傳本機路徑。

<!-- _footer: "模型只供應評語與選擇，不替換計算值" -->

---

<!-- _class: compact -->

## 生成的程式碼永遠不進主機程序

- **容器隔離**：Linux 容器：無主機掛載、無外部網路、有資源上限；以有界資料協定交換 DataFrame
- **程序監督**：Supervisor 管 worker 程序群組；控制用唯一 launch 身分，不信任資料庫裡的 PID
- **可追蹤**：manifest、來源 hash、資料指紋、image ID、random_seed 亂數 context

> **不承諾：** 跨環境完全重現；強殺 supervisor 時，主機 worker 的清理證據可能缺失。

<!-- _footer: "細節見 docs/strategy_contract.md 的 Security Model" -->

---

<!-- _class: compact -->

## 容易出錯的地方，系統先擋一下

| 容易出錯的地方 | 系統怎麼擋 |
| --- | --- |
| **條件沒講清楚** | 先追問標的、期間、K 棒頻率，不自己猜 |
| **還沒同意就開跑** | 先交確認單；沒有明確同意不執行，改了設定要重新確認 |
| **程式或設定被換掉** | 每次執行都綁定固定版本，事後查得到用的是哪一版 |
| **策略程式寫錯** | 開跑前先用假資料試跑，不合格直接擋下 |
| **算不出來就填 0** | 標成「不可用」並寫明原因 |
| **拿不能比的數字比** | 金額、標的或期間不同時，報告標明不可比 |

<!-- _footer: "來源：src/tradingdev/mcp/prompts.py、docs/strategy_contract.md、docs/run_artifacts.md（main f328b8f）" -->

---

<!-- _class: compact -->

## AI 寫的策略程式，只在隔離的容器裡跑

- **主機（可信）**：取得行情、回測引擎、保存結果、內建策略
- **容器（AI 生成的策略）**：讀不到主機檔案 · 連不上網路 · 用量有上限 · 沒有管理權限
- **兩邊只傳純資料**：主機傳行情資料進去，容器傳回交易訊號

| 送進去前 | 執行中 | 用完之後 |
|---|---|---|
| 核對程式內容的指紋，確認是同一份 | 只交換表格資料，格式不對就直接失敗 | 移除容器，確認清乾淨才算結束 |

> **限制：** 這是隔離執行，不等於完整的安全沙箱。

<!-- _footer: "來源：ARCHITECTURE.md「程序控制與清理」「容器執行 adapter」、docs/strategy_contract.md 的 Security Model" -->

---

<!-- _class: compact -->

## 回測績效與排行榜用已保存的結果；統計 API 獨立計算，尚未接入查詢（1/2）

| 範圍 | 已提供的內容與限制 |
| --- | --- |
| **績效管線** | 一般 run performance、年化報酬、Sharpe、日／月損益及 run leaderboard |
| **獨立統計 API** | `domain/statistics`：期間分析、原始資料績效摘要、損益兩平、參數高原與 Sharpe 推論 |
| **整合邊界** | 統計 API 尚未接入績效發布或 MCP 結果查詢，並非排行榜新增欄位 |

<!-- _footer: "來源：上游介紹簡報「績效管線與獨立統計 API」；模組責任與統計慣例見 ARCHITECTURE.md 的績效計算與對話摘要" -->

---

<!-- _class: compact -->

## 回測績效與排行榜用已保存的結果；統計 API 獨立計算，尚未接入查詢（2/2）

- **AI 代理透過 MCP 拿得到**：已保存的 run 指標、比較結果與排行榜：get_run_metrics、compare_runs、leaderboard
- **目前要在平台外處理**：Sharpe 區間、PSR／DSR、參數高原；實例二、三需要時都只能自己算或略過

<!-- _footer: "來源：上游介紹簡報「績效管線與獨立統計 API」；模組責任與統計慣例見 ARCHITECTURE.md 的績效計算與對話摘要" -->

---

<!-- _class: compact -->

## 文件承認的六項架構債

| 限制 | 說明 |
| --- | --- |
| 沒有全面依賴反轉 | app services 直接使用具體的儲存與程序 adapters |
| domain 不全是純計算 | crawlers 會連網路，ML registry 會寫檔案 |
| 最佳化在 worker 協調 | 完整流程尚未收斂到 app 層 |
| ML 與排名未接 CLI | ML 不進回測 CLI／Dashboard；RankingService 只有 MCP |
| 不是跨資源交易 | 檔案與 SQLite 共同綁定身分，但不一起提交 |
| 分頁沒有一致快照 | 跨寫入翻頁可能看到變動中的清單 |

<!-- _footer: "來源：ARCHITECTURE.md；修改這些模組時不能假設 domain 無副作用" -->

---

<!-- header: "Appendix 03　保存設計與實作案例" -->
<!-- _class: chapter -->
<!-- _header: "" -->

## Appendix 03　保存設計與實作案例

保存設計、模組責任與兩個實作案例

---

<!-- _class: compact -->

## SQLite 存索引，檔案存內容，兩者以身分綁定（1/2）

**SQLite：索引與摘要**：核心表 jobs、runs、artifacts、job events；執行計畫與研究筆記各有自己的表；單次回測、walk-forward、最佳化的 job ID 就是 run ID；固定比較由父 job 管理各候選 run。

> **發布前核對：** RunPublisher 確認 job、revision、manifest、來源與績效身分後才登錄結果。檔案與 SQLite 不是同一個交易，hash 只檢查內容一致。

<!-- _footer: "來源：docs/run_artifacts.md 的 Workspace Layout、SQLite Tables、Run Publication" -->

---

<!-- _class: compact -->

## SQLite 存索引，檔案存內容，兩者以身分綁定（2/2）

| workspace 位置 | 保存內容 |
| --- | --- |
| `generated_strategies/` | 不可變的 strategy.py、config.yaml、metadata |
| `runs/<run_id>/` | 一次執行的 manifest 與完整結果 |
| `comparisons/<job_id>/` | 固定比較的共用設定、帳本與行情快照 |
| `execution_plans/` | 確認單文字、HTML 與核准鎖 |
| `reports/`、`research_archives/` | HTML 報告與研究 ZIP |
| `ml_datasets/`、`models/` 等 | 獨立的 ML stores，不用 run 索引 |

<!-- _footer: "來源：docs/run_artifacts.md 的 Workspace Layout、SQLite Tables、Run Publication" -->

---

<!-- _class: compact -->

## src/tradingdev 的主要套件（1/2）

| 位置 | 責任 |
| --- | --- |
| mcp/server.py、tools/、workers/ | 組裝 services、註冊工具；背景工作入口與最佳化流程協調 |
| app/（47 檔）＋ contracts/ | 策略生命週期、計畫與提交、ML 研究、結果查詢；DTO 定義 |
| domain/strategies/ | 策略介面、來源載入、參數解析、訊號契約、6 個內建策略 |
| domain/backtest、validation、optimization | 回測引擎、walk-forward、參數搜尋 |
| domain/performance、statistics | 指標目錄與方向；獨立統計：期間換算、損益兩平、績效摘要、參數高原、Sharpe 推論 |

<!-- _footer: "另有 adapters/cli、clients（Codex app-server）、dashboard 與 shared/ 共用工具" -->

---

<!-- _class: compact -->

## src/tradingdev 的主要套件（2/2）

| 位置 | 責任 |
| --- | --- |
| domain/ml/（7 個子套件） | 特徵、模型、逐棒重訓、驗證、registry、排行榜 |
| domain/data、indicators | 行情抓取與快取、技術指標 |
| adapters/storage、execution | SQLite 與檔案保存；程序啟動、監督、容器執行 |
| adapters/reporting、presentation | HTML 報告、圖表、確認單與對話匯出 |

<!-- _footer: "另有 adapters/cli、clients（Codex app-server）、dashboard 與 shared/ 共用工具" -->

---

<!-- _class: case-dialogue case-transcript case-options case-with-input -->
<!-- _header: "Appendix 03　保存設計與實作案例　／　台股四檔持股比例案例" -->

## 實際對話：找出三組規則的最佳持股比例

操作者輸入

> 透過此專案，在本金3000萬的情況下，持有期間五年，每次交易本金的30%，於超過成本價的20%賣出，低於月均價的10%買入，在台積電、聯發科、聯電與國巨之間找到最佳持股比例　一組只加停損、不限持有天數，另一組把持有上限放寬到 60～120 個交易日，再考慮一組交易準則是波動超過本金5%時就賣出，於價格低於周均價的5%買入，持有期間不超過14天，把這三組找到各自最佳持有比例

AI 代理｜回覆步驟整理（非原文）

> 用 4 個選擇題釐清條件 → 4 份固定比較，共 28 次回測 → 平台外算比例，附 4 份報告

<!-- _footer: "操作者輸入照原文引用；AI 代理的步驟依其回覆整理" -->

---

<!-- _class: compact -->
<!-- _header: "Appendix 03　保存設計與實作案例　／　台股四檔持股比例案例" -->

## 四檔股票、三組規則，先釐清條件再回測（1/2）

**共同條件**：本金 3,000 萬，2021-11～2026-10 共五年；每次交易投入本金的 30%；收盤低於月均價 10% 買入，高於成本 20% 賣出；台積電、聯發科、聯電、國巨；每組各自找出最佳持股比例。

| 組別 | 規則 | 比較 |
| --- | --- | --- |
| 第一組 | 加停損，不限持有天數 | 停損 10%、15%、20% |
| 第二組 | 不停損，設持有上限 | 60、90、120 個交易日 |
| 第三組 | 低於周均價 5% 買；相對成本 ±5% 賣 | 最多持有 10 個交易日 |

<!-- _footer: "來源：使用者實際操作的對話紀錄與伺服器回傳結果（2026-11）" -->

---

<!-- _class: compact -->
<!-- _header: "Appendix 03　保存設計與實作案例　／　台股四檔持股比例案例" -->

## 四檔股票、三組規則，先釐清條件再回測（2/2）

**AI 代理先問了四個會直接改變結果的問題，四題都選建議答案：**

| 「波動超過 5% 就賣」 | 「不超過 14 天」 | 第一組停損幅度 | 第二組要不要停損 |
|---|---|---|---|
| 股價相對成本 ±5%，隔天開盤賣出 | 10 個交易日；周均價用 5 日收盤均線 | 10%、15%、20% 都比較，和上一輪相同 | 不停損，只看持有上限的效果 |

<!-- _footer: "來源：使用者實際操作的對話紀錄與伺服器回傳結果（2026-11）" -->

---

<!-- _class: compact -->
<!-- _header: "Appendix 03　保存設計與實作案例　／　台股四檔持股比例案例" -->

## 4 份固定比較、28 次回測，指標與報告都由伺服器產生（1/2）

| 步驟 | 這次的做法 |
| --- | --- |
| 1 準備 | 沿用第一版策略只換參數；每檔一份 prepare_comparison，含 7 組參數 |
| 2 確認 | 交付確認單後停下，同意才 submit_execution_decision |
| 3 執行 | 4 個 job；每檔 7 組共用同一份行情快照，共 28 個 run |
| 4 查詢 | `compare_runs`、`get_run_metrics`、`get_run_trades`、`get_run_equity`、`get_run_diagnostics` |
| 5 報告 | generate_report：每檔一份離線 HTML，約 93～98 MB |

<!-- _footer: "工具名稱取自 main f328b8f 的 mcp/tools；查詢步驟依各項結果所需資料對應" -->

---

<!-- _class: compact -->
<!-- _header: "Appendix 03　保存設計與實作案例　／　台股四檔持股比例案例" -->

## 4 份固定比較、28 次回測，指標與報告都由伺服器產生（2/2）

| 伺服器內計算 | 平台外由 AI 代理計算 |
| --- | --- |
| 單檔 Sharpe、總報酬、最大回撤、年化報酬與波動 | 持股比例：四檔每日權益依權重合併，取組合 Sharpe 最大 |
| 逐筆進出場價格與損益 | 前後半各自重算比例，檢查穩定度 |
| few_trades 診斷：已平倉交易少於 20 筆 | 加入條件的相關係數推論（伺服器沒有跨股票相關係數） |
| 比較報告的總覽表與各組切換 | — |

<!-- _footer: "工具名稱取自 main f328b8f 的 mcp/tools；查詢步驟依各項結果所需資料對應" -->

---

<!-- _class: compact -->
<!-- _header: "Appendix 03　保存設計與實作案例　／　台股四檔持股比例案例" -->

## 最佳比例都落在台積電與聯發科，聯電與國巨為 0%（1/2）

| 規則 | 台積電／聯發科／聯電／國巨 | Sharpe | 5 年總報酬 | 最大回撤 |
| --- | --- | --- | --- | --- |
| 參考：第一版（不停損、不限天數） | 65／35／0／0 | 0.96 | 25.4% | −8.3% |
| **第一組：停損 20%** | **65／35／0／0** | **0.95** | **23.5%** | **−6.8%** |
| 第一組：停損 10% | 70／30／0／0 | 0.93 | 20.2% | −6.2% |
| 第一組：停損 15% | 55／45／0／0 | 0.55 | 12.8% | −7.1% |

- **第一組**：停損 20% 與 10% 接近第一版，回撤較小；15% 反而最差

<!-- _footer: "本金 3,000 萬；以組合 Sharpe 最大為準；比例在平台外計算，單檔指標來自伺服器" -->

---

<!-- _class: compact -->
<!-- _header: "Appendix 03　保存設計與實作案例　／　台股四檔持股比例案例" -->

## 最佳比例都落在台積電與聯發科，聯電與國巨為 0%（2/2）

| 規則 | 台積電／聯發科／聯電／國巨 | Sharpe | 5 年總報酬 | 最大回撤 |
| --- | --- | --- | --- | --- |
| 第二組：上限 120 日 | 55／45／0／0 | 0.84 | 22.7% | −7.5% |
| 第二組：上限 60 日 | 70／30／0／0 | 0.72 | 16.7% | −7.7% |
| 第二組：上限 90 日 | 60／40／0／0 | 0.64 | 15.2% | −7.3% |
| 第三組：周線短線，10 日 | 0／100／0／0 | −0.17 | −4.7% | −14.0% |

- **第二組**：都比第一版差；上限越寬越接近，上一輪 20 日上限明顯太短
- **第三組**：四檔全虧；每次買賣約 0.7% 成本吃掉 ±5% 的空間

<!-- _footer: "本金 3,000 萬；以組合 Sharpe 最大為準；比例在平台外計算，單檔指標來自伺服器" -->

---

<!-- _class: case-dialogue case-transcript case-options case-with-input -->
<!-- _header: "Appendix 03　保存設計與實作案例　／　台股四檔持股比例案例" -->

## 實際對話：追問聯電、國巨為什麼是 0%

操作者輸入

> 分析聯電/國巨為甚麼都是0/0在各組選擇上，只能透過伺服器內的功能計算

AI 代理｜回覆步驟整理（非原文）

> 只用伺服器的比較指標、交易明細與診斷解釋原因 → **不重算比例；伺服器給不出的部分直接標明**

<!-- _footer: "操作者輸入照原文引用；AI 代理的步驟依其回覆整理" -->

---

<!-- _class: compact -->
<!-- _header: "Appendix 03　保存設計與實作案例　／　台股四檔持股比例案例" -->

## 單檔 Sharpe 每組墊底，加進組合只會拉低 Sharpe（1/2）

| 單檔 Sharpe | 台積電 | 聯發科 | 聯電 | 國巨 |
| --- | --- | --- | --- | --- |
| 第一版 | 0.79 | 0.80 | **0.10** | **0.07** |
| 第一組 停損 10% | 0.80 | 0.73 | **−0.23** | **0.10** |
| 第一組 停損 15% | 0.40 | 0.50 | **−0.29** | **−0.16** |
| 第一組 停損 20% | 0.79 | 0.75 | **−0.21** | **−0.41** |

- **風險相近，報酬接近零**：第一版年化報酬／波動：聯電 0.47%／8.30%、國巨 0.23%／8.48%；聯發科 6.58%／8.36%、台積電 3.59%／4.61%

<!-- _footer: "單檔指標來自伺服器的回測指標；加入條件為 AI 代理在平台外推論，伺服器不提供跨股票相關係數" -->

---

<!-- _class: compact -->
<!-- _header: "Appendix 03　保存設計與實作案例　／　台股四檔持股比例案例" -->

## 單檔 Sharpe 每組墊底，加進組合只會拉低 Sharpe（2/2）

| 單檔 Sharpe | 台積電 | 聯發科 | 聯電 | 國巨 |
| --- | --- | --- | --- | --- |
| 第二組 上限 60 日 | 0.64 | 0.58 | **−0.24** | **−0.12** |
| 第二組 上限 90 日 | 0.50 | 0.56 | **−0.43** | **−0.32** |
| 第二組 上限 120 日 | 0.62 | 0.76 | **−0.51** | **−0.24** |
| 第三組 周線短線 | −0.44 | −0.17 | **−0.22** | **−0.26** |

- **加入組合的條件**：`SR個股 ＞ ρ × SR組合`，只做多、追求組合 Sharpe 時才成立。Sharpe 為負就幾乎不可能；國巨停損 10% 是 0.10，相關係數要低於約 0.125 才有幫助

<!-- _footer: "單檔指標來自伺服器的回測指標；加入條件為 AI 代理在平台外推論，伺服器不提供跨股票相關係數" -->

---

<!-- _class: compact -->
<!-- _header: "Appendix 03　保存設計與實作案例　／　台股四檔持股比例案例" -->

## 獲利只靠一兩筆，跳空讓停損擋不住（1/2）

`few_trades`：伺服器診斷，已平倉交易少於 20 筆

- **聯電（第一版）**：只有 1 筆已平倉：2022-02 買進，持有 963 根 K 棒才到 +20%，淨賺約 262 萬。2026-07 約 141 元買進，期末 111.5 元，帳面淨虧約 191 萬
- **國巨（第一版）**：2 筆已平倉合計淨賺約 443 萬；2026-07 約 915 元買進，期末 502 元，帳面淨虧約 409 萬
- **勝率 100% 的陷阱**：報告只算已平倉交易，期末還沒賣的大額虧損不在勝率裡

<!-- _footer: "交易資料來自伺服器回傳的逐筆進出場價；實際虧損由 AI 代理以進出場價換算" -->

---

<!-- _class: compact -->
<!-- _header: "Appendix 03　保存設計與實作案例　／　台股四檔持股比例案例" -->

## 獲利只靠一兩筆，跳空讓停損擋不住（2/2）

**設定的停損 vs 實際虧損**（收盤判斷、隔天開盤成交，跳空下跌時虧損超過設定）

| 交易 | 停損 | 實際 |
| --- | --- | --- |
| 聯電 2022-02 | 10% | 約 −13.5% |
| 聯電 2026-07 | 10% | 約 −18.5% |
| 聯電 2026-10 | 10% | 約 −18.0% |
| 國巨 2024-08→2025-04 | 20% | 約 −30.1% |
| 國巨 2026-07 | 20% | 約 −27.6% |

<!-- _footer: "交易資料來自伺服器回傳的逐筆進出場價；實際虧損由 AI 代理以進出場價換算" -->

---

<!-- _class: compact -->
<!-- _header: "Appendix 03　保存設計與實作案例　／　台股四檔持股比例案例" -->

## 這次的統計限制，main 上已有哪些功能可以處理（1/2）

| 樣本小 | 選擇偏誤 | 參數不連續 | 穩定度 |
|---|---|---|---|
| 每檔每組 1～25 筆交易，伺服器已標 few_trades | 28 次回測再搜權重，都在同一段歷史挑最佳 | 停損 15% 比 10% 和 20% 都差 | 前後半重算，只有停損 20% 維持 70/30、75/25 |

| 這個案例需要 | main 上的對應功能 | AI 代理能否透過 MCP 使用 |
| --- | --- | --- |
| 0.95 這個 Sharpe 有多不確定 | sharpe_interval_outcome 區間 | 獨立計算，未接 MCP |
| 修正多次搜尋的選擇偏誤 | PSR／DSR | 獨立計算，未接 MCP |
| 參數是否落在穩定區 | 參數高原 | 獨立計算，未接 MCP |

<!-- _footer: "AI 代理建議的下一步：用前段資料挑規則與比例，再用後段資料驗證" -->

---

<!-- _class: compact -->
<!-- _header: "Appendix 03　保存設計與實作案例　／　台股四檔持股比例案例" -->

## 這次的統計限制，main 上已有哪些功能可以處理（2/2）

| 這個案例需要 | main 上的對應功能 | AI 代理能否透過 MCP 使用 |
| --- | --- | --- |
| 和大盤比較 | 依資產類別固定的基準（台股 ^TWII） | 可以：回測附帶基準 |
| 用沒參與挑選的資料驗證 | walk-forward（prepare_walk_forward） | 可以：只驗規則，比例不在內 |
| 持股比例與相關係數 | 上游列為後續方向 | 平台外計算 |

<!-- _footer: "AI 代理建議的下一步：用前段資料挑規則與比例，再用後段資料驗證" -->

---

<!-- _class: compact -->
<!-- _header: "Appendix 03　保存設計與實作案例　／　測試 1、2 重現案例" -->

## 24 次引擎回測，比例組合在平台外試算（1/2）

- **測試 1｜鴻海 2317.TW**：跌破月線買、+20% 停利；四個計畫只改每筆投入金額（A／B／C 固定名目，D 全額複利），共 4 次回測
- **測試 2｜台積電、鴻海、聯發科、0050**：5 種出場 × 4 檔共 20 次回測；再試算 286 種比例 × 5 = 1,430 組，每年再平衡

| 步驟 | 這次的實作 |
| --- | --- |
| 策略 | 依會議截圖重寫成生成策略，保存成固定 revision |
| 證據 | 記錄參數來源：20 日與 +20% 為使用者提供，60／120 日為模型提案 |
| 準備與確認 | 每個計畫一張確認單，附確認前程式檢查；同意後才提交 |

<!-- _footer: "共同設定：Yahoo 日 K、2021-11-09～2026-11-06、本金 3,000 萬、手續費 0.29%、滑價 0.05%、年化 252 日" -->

---

<!-- _class: compact -->
<!-- _header: "Appendix 03　保存設計與實作案例　／　測試 1、2 重現案例" -->

## 24 次引擎回測，比例組合在平台外試算（2/2）

| 步驟 | 這次的實作 |
| --- | --- |
| 結果 | 指標取自各 run 的 result.json；每日權益供平台外合併 |
| 報告 | 測試 1 四計畫、測試 2 原版四檔各一份離線報告：配對交易、大盤基準、可比性標記 |
| 平台外(AI 代理自行計算) | 對齊交易曆、1,430 組比例、每年再平衡與成本、組合 Sharpe |

<!-- _footer: "共同設定：Yahoo 日 K、2021-11-09～2026-11-06、本金 3,000 萬、手續費 0.29%、滑價 0.05%、年化 252 日" -->

---

<!-- _class: compact -->
<!-- _header: "Appendix 03　保存設計與實作案例　／　測試 1、2 重現案例" -->

## 確認單分清楚：哪些由後端固定，哪些由模型提供（1/2）

| 區塊 | 這張確認單寫了什麼 |
| --- | --- |
| 構想與規則 | 公式、變數、暖機與狀態轉移；標明由模型提供，後端不證明符合 |
| 參數來源 | 20 日與 +20%：使用者提供；60／120 日：模型提案，未經查證 |
| 執行語意 | 訊號延後一根、開盤成交；停利 20%；期末依收盤強制平倉 |

> **閱讀確認單不會啟動回測：** 要回到對話明確同意；改了程式或設定就要重新檢查、重新確認。

<!-- _footer: "來源：測試 2「台積電、最長持有 60 日」的回測前確認單（計畫 c8765879…）" -->

---

<!-- _class: compact -->
<!-- _header: "Appendix 03　保存設計與實作案例　／　測試 1、2 重現案例" -->

## 確認單分清楚：哪些由後端固定，哪些由模型提供（2/2）

| 區塊 | 這張確認單寫了什麼 |
| --- | --- |
| 成本與年化 | 手續費 0.29%、滑價 0.05%；年化 252 日為明示設定 |
| 確認前檢查 | 5 項通過；合成資料 80 與 240 根；約 26 秒 |
| 技術識別 | 確認計畫、內容摘要、策略 revision、研究證據摘要 |

**確認單自己寫出的限制**：停損用引擎設定，持有上限在策略碼內計數，出場語意不對稱；比例組合與再平衡在專案外計算，有事後挑選偏差；費率簡化，Yahoo 價格未還原股利；沒有對方原始碼，是重寫不是逐筆重播。

<!-- _footer: "來源：測試 2「台積電、最長持有 60 日」的回測前確認單（計畫 c8765879…）" -->

---

<!-- _class: compact -->
<!-- _header: "Appendix 03　保存設計與實作案例　／　測試 1、2 重現案例" -->

## 三種不停損設定幾乎打平，兩組停損明顯較差（1/2）

| 出場設定 | 最佳比例 台／鴻／聯／0050 | Sharpe | 五年獲利 | 年化 | 最大回撤 |
| --- | --- | --- | --- | --- | --- |
| 原版（無停損） | 40／10／0／50 | 1.033 | 4,710 萬 | 21.7% | 37.0% |
| 停損 −10% | 40／0／0／60 | 0.699 | 1,429 萬 | 8.4% | 18.8% |
| 停損 −15% | 100／0／0／0 | 0.816 | 2,933 萬 | 15.2% | 26.2% |
| 持有上限 60 日 | 50／0／0／50 | 1.036 | 5,596 萬 | 24.5% | 38.4% |
| **持有上限 120 日** | **0／0／0／100** | **1.040** | **4,728 萬** | **21.8%** | **36.8%** |

<!-- _footer: "每種出場各從 286 種比例（10% 一格）選 Sharpe 最高者；比例與組合指標在平台外計算" -->

---

<!-- _class: compact -->
<!-- _header: "Appendix 03　保存設計與實作案例　／　測試 1、2 重現案例" -->

## 三種不停損設定幾乎打平，兩組停損明顯較差（2/2）

![h:200](assets/case3-sharpe-by-exit.png)

- **前幾名差距很小**：1,430 組中 23 組與最佳相差不到 0.02；截圖的 10/20/0/70 在原版排第 29（0.987）
- **只看獲利**：持有 60 日全押台積電賺 6,861 萬，但回撤 43.1%、Sharpe 1.006

<!-- _footer: "每種出場各從 286 種比例（10% 一格）選 Sharpe 最高者；比例與組合指標在平台外計算" -->

---

<!-- _class: compact -->
<!-- _header: "Appendix 03　保存設計與實作案例　／　測試 1、2 重現案例" -->

## 單檔差異很大，停損讓鴻海與聯發科轉為虧損（1/2）

| Sharpe／交易數 | 原版 | 停損 −10% | 停損 −15% | 持有 60 日 | 持有 120 日 |
| --- | --- | --- | --- | --- | --- |
| 台積電 | 0.967／7 | 0.640／7 | 0.816／7 | **1.006／19** | 0.905／12 |
| 鴻海 | 0.630／6 | −0.113／6 | −0.016／6 | 0.480／20 | 0.493／13 |
| 聯發科 | 0.377／5 | **−1.071／5** | **−1.341／5** | 0.356／19 | 0.506／14 |
| 0050 | 0.956／7 | 0.645／7 | 0.216／7 | 0.990／19 | **1.040／13** |

<!-- _footer: "數值取自各 run 已保存的 result.json（get_run_metrics）；2021-11-09～2026-11-06" -->

---

<!-- _class: compact -->
<!-- _header: "Appendix 03　保存設計與實作案例　／　測試 1、2 重現案例" -->

## 單檔差異很大，停損讓鴻海與聯發科轉為虧損（2/2）

- **停損傷害最大的是聯發科**：兩組停損各 5 筆交易全虧，勝率 0%；原版同樣 5 筆、勝率 80%
- **持有上限增加交易數**：原版 5–7 筆；上限 60 日增為 19–20 筆，結果仍依標的而異
- **樣本仍然太小**：依診斷規則門檻（交易數 < 20），20 次中有 19 次會被標 few_trades

<!-- _footer: "數值取自各 run 已保存的 result.json（get_run_metrics）；2021-11-09～2026-11-06" -->

---

<!-- _class: compact -->
<!-- _header: "Appendix 03　保存設計與實作案例　／　測試 1、2 重現案例" -->

## 五年獲利 4,728 萬，但第一年先虧 26%（1/2）

![h:235](assets/case3-best-equity.png)

<!-- _footer: "曲線由附檔最佳組合權益.npy 重繪；Sharpe 為日報酬平均 ÷ 標準差 × √252" -->

---

<!-- _class: compact -->
<!-- _header: "Appendix 03　保存設計與實作案例　／　測試 1、2 重現案例" -->

## 五年獲利 4,728 萬，但第一年先虧 26%（2/2）

| 年度起 | 2021-11 | 2022-11 | 2023-11 | 2024-11 | 2025-11 |
| --- | --- | --- | --- | --- | --- |
| 損益 | **−786 萬** | 535 萬 | 1,659 萬 | 936 萬 | 2,383 萬 |
| 報酬 | **−26.2%** | 24.2% | 60.4% | 21.2% | 44.6% |

持有上限 120 日、100% 0050：Sharpe 1.040、年化 21.8%、最大回撤 36.8%

> **平台外的算法：** 四檔交易曆取聯集 1,212 根（交集 1,211 根檢查差異 < 0.001）；權益除以 3,000 萬合併；每年調回比例，成本 Σ|Δ權重| × 價值 × 0.29%

<!-- _footer: "曲線由附檔最佳組合權益.npy 重繪；Sharpe 為日報酬平均 ÷ 標準差 × √252" -->

---

<!-- _class: compact -->
<!-- _header: "Appendix 03　保存設計與實作案例　／　測試 1、2 重現案例" -->

## 投入金額放大報酬，也放大回撤，Sharpe 幾乎不動（1/2）

| 計畫 | 每筆投入 | 總損益 | 總報酬 | 年化 | 最大回撤 | Sharpe | 會議截圖（參考） |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **A** | 1,000 萬 | 780 萬 | 26.01% | 4.93% | 14.03% | 0.635 | 909 萬／+30% |
| **B** | 2,000 萬 | 1,561 萬 | 52.02% | 9.11% | 24.07% | 0.660 | 1,818 萬／+61% |
| **C** | 2,900 萬 | 2,263 萬 | 75.43% | 12.41% | 30.94% | 0.676 | 2,635 萬／+88% |
| **D** | 全額複利 | 2,986 萬 | 99.52% | 15.46% | 48.86% | 0.630 | 3,661 萬／+122% |

<!-- _footer: "測試 1 引擎報告（SHA-256 02b24d1f…）；圖為計畫 A（藍）與 D（橘）的回撤；四個計畫只改 backtest.position_size" -->

---

<!-- _class: compact -->
<!-- _header: "Appendix 03　保存設計與實作案例　／　測試 1、2 重現案例" -->

## 投入金額放大報酬，也放大回撤，Sharpe 幾乎不動（2/2）

![h:200](assets/case3-drawdown-a-vs-d.png)

- **A、B、C 損益正好 1 : 2 : 2.9**：六筆進出場完全相同，固定名目不複利，損益和每筆投入同比例；Sharpe 只在 0.630–0.676 之間
- **唯一一筆虧損是期末強制平倉**：2026-06-11 以 260.13 買進，回測最後一天以 224.39 平倉並扣出場費；A 淨虧 143 萬、D 淨虧 994 萬

<!-- _footer: "測試 1 引擎報告（SHA-256 02b24d1f…）；圖為計畫 A（藍）與 D（橘）的回撤；四個計畫只改 backtest.position_size" -->

---

<!-- _class: compact -->
<!-- _header: "Appendix 03　保存設計與實作案例　／　測試 1、2 重現案例" -->

## 報告附上大盤基準，也標出哪些數字不能直接比（1/2）

| 原版策略 | 累積報酬 | 年化 | 最大回撤 | Sharpe |
| --- | --- | --- | --- | --- |
| 台積電 2330 | **187.29%** | 24.56% | 45.68% | **0.97** |
| 鴻海 2317 | 99.52% | 15.46% | 48.86% | 0.63 |
| 聯發科 2454 | 40.58% | 7.34% | 54.37% | 0.38 |
| 元大台灣50 0050 | **139.23%** | 19.88% | 36.38% | **0.96** |
| **^TWII 買進持有** | **124.80%** | **18.38%** | **31.63%** | **0.88** |

**對照大盤：** 只有台積電、0050 的報酬與 Sharpe 高於 ^TWII 買進持有；四檔的最大回撤都比大盤深

<!-- _footer: "測試 2 引擎報告（SHA-256 ee399c18…）；基準 tw_equity_taiex_buy_and_hold：首根收盤買進、套用同樣成本、價格指數不含股利" -->

---

<!-- _class: compact -->
<!-- _header: "Appendix 03　保存設計與實作案例　／　測試 1、2 重現案例" -->

## 報告附上大盤基準，也標出哪些數字不能直接比（2/2）

- **期間差一根 K 棒，就不算超額報酬**：^TWII 只對齊 1210／1211 根（0050 為 1211／1212），報告保留兩邊數值，差異欄顯示「—」
- **金額指標標為不可比**：`compare_runs` 的 `metric_compatibility`：投入金額或標的不同時，總損益、回撤金額不可比；報酬率、Sharpe 仍可比
- **LLM 評語不算計算結果**：觀察、原因假說與截圖對照都標為「非計算結果」，後端不驗證結論，也不當成回測證據

<!-- _footer: "測試 2 引擎報告（SHA-256 ee399c18…）；基準 tw_equity_taiex_buy_and_hold：首根收盤買進、套用同樣成本、價格指數不含股利" -->

---

<!-- _class: references -->

## 參考資料

- [軟體需求規格（SRS）](https://github.com/Kuan-Lun/tradingdev/blob/main/docs/requirements.md)
- [MCP server 註冊與服務組裝](https://github.com/Kuan-Lun/tradingdev/blob/main/src/tradingdev/mcp/server.py)
- [工具模組（strategy / execution / data / jobs / runs / history / artifacts / research_notes / feature_requests / strategy_cleanup / ml）](https://github.com/Kuan-Lun/tradingdev/blob/main/src/tradingdev/mcp/tools/)
- [SERVER_INSTRUCTIONS 九步工作流](https://github.com/Kuan-Lun/tradingdev/blob/main/src/tradingdev/mcp/prompts.py)
- [系統架構](https://github.com/Kuan-Lun/tradingdev/blob/main/ARCHITECTURE.md)

---

## 王冠倫的心得 (1/2)

- 合作專案為完善原已發布的 TradingDev
  - 經由定期會議可以讓我確認從前制定的暫時方案是否正確、業界比較常見的做法或特殊考量
- 了解挑選 LLM 模型的性質
  - 本專案開發期間會需要多次使用 LLM 來模擬使用者操作來完成整合測試，使用高階模型會錯估成功率 (使用者的 AI 代理模型品質不可控)
  - 由提示詞驅動的文件檢查需要分流以此避免 Token 過度消耗
  - 設定 AI 審查工具自動調用的參附資料範圍以此增加審查品質與降低 Token 消耗

---

## 王冠倫的心得 (2/2)

- 學習新的 AI 自動開發流程：
  - ISO/IEC/IEEE 29148:2018 確保了需求層級
    - 「規格」可以由無 MCP 伺服器時的本地思考過程抽取
    - 「需求」可以轉化成開發提示詞
    - 「驗收」可以由有無 MCP 伺服器運行時的執行差異做為基準
  - 自動發布 Issue -> 喚醒 LLM 開發後發布 PR -> 排序檢查後集中合併

---

<!-- _class: compact -->

## 許竣鈞的心得

這次專案讓我學到，理論和程式碼對不上的時候，常常該做的是補充說明而不是改演算法。例如 deflated Sharpe ratio 在理論上該用「有效試驗數」，但格點搜尋的試驗彼此高度相關，有效數要靠報酬相關性分群才估得出來，超出這次的範圍。所以程式用實際的試驗數，並在文件裡寫明這是刻意偏保守的設計選擇、而不是已經證實的性質——把限制說清楚，而不是假裝它不存在。

也感謝學長能在我卡住check full and pytest，只能直接提出PR時，能幫我進行程式碼檢測，還有對於我的PR提出修改建議，也謝謝PM在每次會議時的主持，不會讓我們的方向跑掉。

總而言之，有幸能夠透過這次專案，把課堂上的理論與實務上的操作進行對接，也感謝PM 與學長們的鼎力支持，才能完整的完成這次合作。

<!-- _footer: "許竣鈞 國立成功大學經濟學系 學士班（雙主修資訊工程學系）" -->

---

<!-- _class: compact -->

## Morten Weng 的心得

這次專案讓我學到軟體測試很重要，尤其在關於ai程式的驗收，例如檢查每一個 Import（AI 會幻覺出不存在的套件），還有預期結果 (Expected values) 絕對不能來自當前程式的輸出。

最重要的是 Validation（驗證系統是否做對的事）永遠不能外包給 AI。

感謝永豐黃如君以及兩位隊友們的幫助讓我得以完成這個有意義的專案。

<!-- _footer: "Morten Weng University of Bergen 博士（應屆畢業）" -->
