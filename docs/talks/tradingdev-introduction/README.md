# TradingDev 簡報編輯指南

簡報內容與講者稿維護在 [slides.md](slides.md)，版型維護在 [theme.css](theme.css)；
[index.html](index.html) 是產生後提交與發布的瀏覽器版本。
大綱、章節整理與後續目標集中放在 `slides.md`，本文件只說明編輯與發布操作。

## 編輯與預覽

首次使用先依[開發指南](../../development.md#簡報的編輯與輸出)初始化環境與 Marp 工具鏈，
再用 VS Code 開啟整個 repository 資料夾。

1. 開啟命令選擇區，執行 **Tasks: Run Task → slides: watch**。
2. 編輯並儲存 `slides.md` 或 `theme.css`，工作會自動更新同目錄的 `index.html`。
3. 用瀏覽器開啟 `index.html`；每次儲存後重新整理瀏覽器查看結果。
4. 結束時執行 **Tasks: Terminate Task → slides: watch**。

編譯訊息可在 `slides: watch` 的專用終端機查看。只需單次輸出時，執行
**Tasks: Run Task → slides: build**。自動啟動監看、命令列操作與本機 HTTP
預覽方式見[開發指南](../../development.md#簡報的編輯與輸出)。
提交簡報內容或版型修改時，一併提交重新產生的 `index.html`。

瀏覽器版內含樣式與播放程式，可離線開啟。使用左右方向鍵換頁，`F` 切換全螢幕，
`P` 開啟講者模式；講者模式須允許新視窗，本機檔案模式有通訊限制時改用 HTTP 預覽。

## 章節與子主題版型

- **章首頁**使用 `chapter`：深色底、章節編號與主題說明。
- **子主題起始頁**使用 `subsection`：沿用淺色底，標題上方以綠色短線標示轉場。
  頁首用 `header` 標出章節與子主題，例如 `02　策略研究　／　02.2　策略實驗與結果`。
- **延伸內容頁**沿用相同 `header`，不加 `subsection`，依內容選用表格、列表或引言。
  下一個子主題或章節開始時更新 `header`，小結頁則改回章節名稱。

子主題起始頁同時呈現該主題的第一個重點，不必額外插入分隔頁。概念引言可搭配
`statement`，如「這裡的『研究能力』」與「自己的規則也能成為策略」；表格較密時可搭配
`compact`。這兩個 class 調整內容呈現，不代表章節層級，`statement` 也可用於小結。

只有需要區分的主題群才設子主題，連續的短章節可直接展開。後續開發方向沿用
`TODO-01` 等既有識別碼作為子主題標示；同一目標的補充頁沿用其頁首。
新增或移動頁面時，同步核對閱讀路線及其他 `index.html#頁碼` 連結。

## 透過 Cloudflare 自動發布

Cloudflare Workers Builds 可連接本 repository，在 `main` 更新時發布已提交的
`index.html`。本機仍須先編譯、檢查並提交 HTML；雲端只複製與部署成品，
不執行 Marp、Python 檢查或安裝 TradingDev 產品依賴。

在 Cloudflare 的 **Workers & Pages → Create application → GitHub** 選擇本
repository；GitHub 授權範圍只選需要發布的 repository。設定如下：

| 欄位 | 設定 |
| --- | --- |
| Worker name | 與根目錄 [wrangler.jsonc](../../../wrangler.jsonc) 的 `name` 相同。 |
| Production branch | `main`。 |
| Root directory | repository 根目錄。 |
| Build command | 下方的複製指令。 |
| Deploy command | `npx wrangler deploy`。 |
| Preview command（非正式分支） | 啟用分支預覽時使用 `npx wrangler preview`。 |

Cloudflare Workers Builds 的 `Preview command` 契約說明此命令建立或更新預覽，
並產生預覽網址，原文節錄：

> It defaults to `npx wrangler preview`, which creates or updates a Preview and produces a Preview URL.

Build command：

```bash
mkdir -p site-dist/tradingdev && cp docs/talks/tradingdev-introduction/index.html site-dist/tradingdev/index.html
```

在 **Settings → Build → Build Variables and Secrets** 加入
`SKIP_DEPENDENCY_INSTALL=1`，停用自動安裝專案依賴。部署仍會使用 Wrangler；
Workers Builds 可自動建立部署 token，不需將 token 放進 repository。
`site-dist/` 是忽略的發布產物，只包含要公開的 HTML；不會將整個 `docs/` 發布。

分支預覽使用 `wrangler.jsonc` 的 `previews` 區塊；即使沒有獨立設定，也須保留空物件。
Cloudflare 的 Wrangler 預覽設定契約明定此要求，原文節錄：

> The `previews` block is required, but it can be empty if your Preview does not need separate settings.

靜態資源仍使用頂層 `assets`，正式網域的 `routes` 不放入 `previews`。
在建置記錄確認分支與 commit，成功後開啟產生的預覽網址並加上 `/tradingdev/` 查看簡報。
分支預覽與 `main` 正式部署是兩次獨立建置，須分別核對結果；本機簡報檢查不代表雲端部署成功。

若建置設定顯示 Git 帳戶中斷連線，按 Git repository 旁的 **Manage**，在 GitHub
完成必要的重新驗證，確認 Cloudflare 應用程式仍可存取本 repository，再回到
Cloudflare 重新整理設定頁。確認連線警告消失、Production branch 為 `main`，
且 **Disable builds** 關閉。下一次主線更新後，核對建置記錄的 commit 與部署結果；
手動重試建置成功不代表 GitHub 自動觸發已驗證。

### 共用網域與專案路徑

`slides.klwang.tw` 是多個專案共用的簡報網域。Custom Domain 綁定獨立的
`klwang-slides` 入口 Worker，由管理者在 Cloudflare 維護；本 repository
只部署 `klwang-slides-tradingdev`，不接管整個網域。

根目錄的 `wrangler.jsonc` 宣告本專案的兩條 Route：

- `slides.klwang.tw/tradingdev`：不帶尾端斜線的入口。
- `slides.klwang.tw/tradingdev/*`：簡報及其子路徑。

Route 優先於共用入口的 Custom Domain；路由保留原始路徑，因此發布檔案仍放在
`site-dist/tradingdev/index.html`。不要改成 `slides.klwang.tw/*` 或將整個
Custom Domain 綁回本專案，也不要使用會涵蓋 `/tradingdev-other` 的
`/tradingdev*`。不帶尾端斜線且附有 query 的網址由共用入口補上斜線後轉址。

首次設定時，管理者先在 **Workers & Pages → klwang-slides → Domains →
Add domain** 選擇 `klwang.tw`，在 Subdomain 填入 `slides`，啟用於 Production。
網域須屬於同一 Cloudflare 帳號的有效 zone，Cloudflare 負責 DNS 與 HTTPS。
TradingDev 的部署權限須包含該 zone 的
Workers Routes 編輯權限；部署後在本專案 Worker 的 **Domains** 核對兩條 Route。
沿用舊設定時，先建立上述 Route，再將 Custom Domain 移到共用入口，以保留簡報入口。

新增其他專案時，使用獨立 Worker 與不同的路徑前綴，各自連接自己的 repository；
不覆寫本專案 Worker，也不移動共用 Custom Domain。共用入口只將根路徑 `/` 以
302 轉向[演講與簡報目錄](https://www.klwang.tw/talks/)，由管理者另外維護，
不隨 TradingDev 發布更新。

簡報位址為 [https://slides.klwang.tw/tradingdev/](https://slides.klwang.tw/tradingdev/)。只有部署成功才會更新網站；
repository 中的設定檔不代表已連接 Cloudflare 或發布。

## 透過 GitHub Pages 發布

GitHub 的一般檔案頁只會顯示 HTML 原始碼。若要直接播放，可使用同一 repository
的 GitHub Pages。

取得發布授權並透過 PR 合併至主線後，由管理者在
**Settings → Pages → Deploy from a branch** 選擇主線與 `/docs`。
這會發布 `/docs` 下的內容，`.nojekyll` 讓 GitHub 直接提供靜態檔案；
已有 Pages 網站時，先確認原有發布設定。

沒有自訂網域且使用目前 repository 名稱時，預期位址為：

```text
https://kuan-lun.github.io/tradingdev/talks/tradingdev-introduction/
```

此位址不代表已發布；完整設定見
[GitHub Pages 官方說明](https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site)。
