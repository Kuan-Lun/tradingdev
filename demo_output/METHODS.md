# 展示圖方法說明（METHODS）

本文件記錄 `demo_output/` 三張圖：`plateau_demo.png`、`kd_equity.png`、
`comparison_forest.png`，以及附帶的 `kd_inference.md`。文件說明其中每個數字的
資料來源、公式與對應的程式碼位置。

所有內容都是從下列程式碼及 pickle 實際讀出的，不是依描述推測：

- `scripts/demo_plateau.py`（已 commit `fcebb68`）
- `scripts/demo_statistics.py`（**未納入 git**）
- `src/tradingdev/domain/statistics/{periods,performance,sharpe,breakeven,plateau,comparison}.py`
- `workspace/data/processed/cache/cd03bf19459b48af.pkl`
- 合併前（`6826e41`）的 `src/tradingdev/domain/backtest/signal_engine.py` 與 `metrics.py`

實作與教科書寫法不同的地方，以 **⚠ 差異** 標註。

---

## 一、資料來源

### 1. `plateau_demo.png`：合成資料

產生程式：`scripts/demo_plateau.py:51-66`（`build_cases`）。這份資料完全不經過回測。

**參數軸（實際值）**，定義於 `scripts/demo_plateau.py:36-37`：

| 軸 | 索引 | 值 |
| --- | --- | --- |
| x：`lookback` | $i = 0,\dots,4$ | 10, 20, 30, 40, 50 |
| y：`threshold` | $j = 0,\dots,4$ | 0.01, 0.02, 0.03, 0.04, 0.05 |

**亂數**：`rng = np.random.default_rng(0)`（`SEED = 0`）。以固定順序抽兩個 $5\times5$ 矩陣：

1. $\varepsilon^{A}_{ij} \sim \mathcal N(0,\,0.05^2)$：`rng.normal(0.0, 0.05, (5, 5))`
2. $\varepsilon^{B}_{ij} \sim \mathcal N(0,\,0.3^2)$：`rng.normal(0.0, 0.3, (5, 5))`

`normal` 的第二個參數是標準差，所以變異數是 $0.05^2$ 與 $0.3^2$。

**案例 A（真平原）**

$$
m^{A}_{ij} = 1.5 - 0.3\,|i-2| - 0.3\,|j-2| + \varepsilon^{A}_{ij}
$$

**案例 B（假尖峰）**

$$
m^{B}_{ij} =
\begin{cases}
2.0 & (i,j) = (1,3) \\
\varepsilon^{B}_{ij} & \text{其他}
\end{cases}
$$

實作細節：$\varepsilon^{B}_{1,3}$ 仍然會被抽出來，只是被 2.0 覆蓋掉，所以不影響其他格子的亂數。

尖峰 $(i,j)=(1,3)$ 對應的參數是 lookback = 20、threshold = 0.04。
`scripts/demo_statistics.py:89-100` 舊版的抽樣順序和公式與此相同，所以兩支腳本的
統計數字一樣，差別只在軸標籤：舊版用的是網格索引 0 到 4。

### 2. `kd_equity.png`：真實回測（讀取快取 pickle）

`latest_cache()`（`scripts/demo_statistics.py:259-263`）依**修改時間**挑選快取目錄中最新的
`.pkl`。目前挑到的是 `cd03bf19459b48af.pkl`。另一個 `e9d1e13c2eb25909.pkl` 的 config、
權益曲線與交易紀錄完全相同，只是時間較舊。

| 項目 | 值 | 來源 |
| --- | --- | --- |
| pickle 檔名 | `workspace/data/processed/cache/cd03bf19459b48af.pkl`（mtime 2026-10-05 06:33 UTC） | 檔案系統 |
| 物件型別 | `PipelineResult`，`mode = "simple"`，`fold_results = []` | pickle |
| 策略 id | `kd_crossover`（version 1.0.0） | `config_snapshot.strategy` |
| 策略類別 | `KDStrategy`（`src/tradingdev/domain/strategies/bundled/kd_strategy/strategy.py`） | 同上 |
| 參數 | `k_period=14, d_period=3, smooth_k=3, overbought=80.0, oversold=20.0` | 同上 |
| 交易對 | `BTC/USDT` | `config_snapshot.backtest.symbol` |
| 時間框架 | `1h` | `config_snapshot.backtest.timeframe` |
| 起訖日期（config） | `2024-01-01` 至 `2024-12-31` | `config_snapshot.backtest` |
| 實際第一根／最後一根 K 棒 | `2024-01-01 00:00 UTC` / `2024-12-31 00:00 UTC` | `backtest_result.timestamps` |
| K 棒根數 | 8,761（相鄰兩根間隔都是 1 小時，沒有缺口） | `len(equity_curve)` |
| 交易筆數 | 412 | `len(trades)` |
| fees | 0.0006 | `config_snapshot.backtest.fees` |
| slippage | 0.0005 | `config_snapshot.backtest.slippage` |
| init_cash | 10,000.0 | `backtest_result.init_cash` |
| 資料來源 crawler | `binance_vision`（**推定**） | 見下方說明 |
| market_type | `futures/um`（**推定**） | 見下方說明 |
| 回測引擎 | `signal` 模式（vectorbt `Portfolio.from_signals`） | `backtest_result.mode` |

**資料來源與 market_type 是推定的**：pickle 的 `config_snapshot.data` 只有
`requirements.market = {symbol, timeframe}`，沒有記錄 `source` 或 `market_type`。
合併前後的 `src/tradingdev/domain/data/schemas.py:50-51` 預設值都是
`source = "binance_vision"`、`market_type = "futures/um"`，
`src/tradingdev/domain/data/requirements.py:13` 也是 `binance_vision`。
因此推定使用這兩個預設值，但 pickle 本身無法證明這一點。

另外，這份 snapshot 也**沒有** `periods_per_year`。目前的
`kd_strategy/config.yaml` 有 `periods_per_year: 365.0`，表示 pickle 是用較舊的 config 產生的。
這個欄位不影響圖中數字，因為腳本固定使用 `DAILY_PERIODS_PER_YEAR = 365.0`。

**產生 pickle 的引擎（合併前 `6826e41`）**：

- `vbt.Portfolio.from_signals(..., price=open_, fees=0.0006, slippage=0.0005)`：
  以 open 價成交，fees 由 vectorbt 從現金扣除並計入 PnL，slippage 以百分比調整成交價。
- `_extract_trades` 把每筆交易的 `fee` 寫死為 `0.0`，`net_pnl = gross_pnl` 等於 vectorbt
  已扣除 fees 的 PnL。`size_quote` 等於 `Size × Avg Entry Price`，也就是已含 slippage 的進場名目金額。
- `total_volume` 等於 $\sum |\text{Size}|\times\text{Entry Price}$，**只計算進場名目金額**。
  以 pickle 實測：`total_volume` = 2,507,242.31，正好等於 $\sum$ `size_quote`。
  若把出場也算進去，則是 5,014,735.72。

### 3. `comparison_forest.png`：真實回測（**重新執行**，不讀 pickle）

`run_kd_variants()`（`scripts/demo_statistics.py:411-420`）載入
`src/tradingdev/domain/strategies/bundled/kd_strategy/config.yaml`，深拷貝後只覆蓋三組參數，
再用 `BacktestService().run_raw_config(config)` 重新回測。

| 標籤 | k_period | d_period | smooth_k | overbought / oversold |
| --- | ---: | ---: | ---: | --- |
| KD(9, 3, 3) | 9 | 3 | 3 | 80.0 / 20.0（沿用 config） |
| KD(14, 3, 3) | 14 | 3 | 3 | 80.0 / 20.0 |
| KD(21, 5, 5) | 21 | 5 | 5 | 80.0 / 20.0 |

其他設定都來自 `config.yaml`：`BTC/USDT`、`1h`、`2024-01-01` 至 `2024-12-31`、
`init_cash = 10000.0`、`fees = 0.0006`、`slippage = 0.0005`。
config 沒有指定 `source` 與 `market_type`，所以使用預設的 `binance_vision` 與 `futures/um`
（`src/tradingdev/app/data_service.py:76`、`schemas.py:50-51`）。

這三次回測的 K 棒根數與交易筆數沒有存檔，pickle 也不包含這些資訊，因此本文件無法列出。
其中 KD(14, 3, 3) 的參數與 pickle 相同，所以預期和 pickle 一樣是 8,761 根 K 棒。

**重要**：`demo_output/comparison_forest.png` 與 `kd_equity.png` 的檔案時間是
2026-10-11 14:43 UTC，**早於**合併上游的 commit `b4aae8d`（14:49 UTC）。
所以目前這張 forest 圖是用**合併前**的引擎產生的。合併後重跑會改用新引擎，
數字可能不同，詳見第五節。依目前程式碼，`BacktestService` 不會寫入
`processed/cache/*.pkl`，所以重跑不會改變 `latest_cache()` 的選擇。

---

## 二、計算公式

符號定義：$E_t$ 為第 $t$ 根 K 棒的權益，$E_0$ 為權益曲線的第一個值。

### 2.1 日頻報酬（`src/tradingdev/domain/statistics/periods.py`）

**UTC 界線**（`periods.py:1-20, 36-45`）：

- 一天是半開區間 $[00{:}00, 24{:}00)$ UTC，一個月是從當月 1 日 00:00 UTC 到下月 1 日 00:00 UTC。
- 時間戳**依原標籤**歸入所屬期間。Binance 以 K 棒的開盤時間作為標籤，所以 23:00 那根
  K 棒算在它開盤的那一天。
- 沒有時區的時間戳視為 UTC，有時區的則先轉成 UTC。

**期末權益**（`periods.py:74-87`）：`resample("D").last().dropna()`，取當期最後一個觀測值。

$$
C_d = E_{\max\{t\,:\,\text{ts}_t \in d\}}
$$

**第一期的基準**（`periods.py:128-130`，`_opening_equity`）：
`closing.shift(1).fillna(E_0)`，所以第一期的期初權益是權益曲線的第一個值 $E_0$，
不是第一天的開盤價。

$$
O_{d_1} = E_0, \qquad O_{d_k} = C_{d_{k-1}}\ (k \ge 2)
$$

**報酬與損益**（`periods.py:90-120`）：

$$
\text{PnL}_d = C_d - O_d, \qquad r_d = \frac{C_d}{O_d} - 1
$$

複利連乘的結果是 $\prod_d (1+r_d) = E_{\text{last}}/E_0$。只要有任何 $E_t \le 0$，
計算報酬時就會 raise。

**缺資料期間**：沒有觀測值的期間會被 `.dropna()` **直接略過，不補 0**。
⚠ **差異**：如果中間有缺口，缺口後第一期的報酬會涵蓋整個缺口期間，
而不是把缺口那幾天記為 0 報酬。本資料 8,761 根 K 棒沒有任何缺口，所以不受影響。

**本資料的實際結果**：最後一根 K 棒是 `2024-12-31 00:00`，所以 2024-12-31 這一天
**只有 1 根 K 棒**，它的日報酬實際上只是 12-30 收盤到 12-31 00:00 這 1 小時的變化。
得到的日數是 366 天，月數是 12 個月。

`kd_inference.md` 的「月報酬為負比例」等於 $\frac{1}{M}\sum_m \mathbf 1[\text{PnL}_m < 0]$，
使用 `period_pnl(..., "monthly")`（`scripts/demo_statistics.py:271`）。

### 2.2 年化 Sharpe（`src/tradingdev/domain/statistics/sharpe.py:69-81, 189-194`）

$$
\widehat{SR}_{\text{period}} = \frac{\bar r}{s}, \quad
s = \sqrt{\frac{1}{n-1}\sum_{d}(r_d-\bar r)^2}, \qquad
\widehat{SR}_{\text{ann}} = \widehat{SR}_{\text{period}}\cdot\sqrt{P}
$$

- $P$ 即 `periods_per_year`，實際值為 **365.0**
  （`scripts/demo_statistics.py:68` 的 `DAILY_PERIODS_PER_YEAR`）。
- 無風險利率固定為 0，標準差使用 `ddof=1`。
- $n<2$ 或所有報酬都相等（`np.ptp == 0`）時回傳 `None`。

⚠ **差異**：

1. 無風險利率為 0，不是扣除 $r_f$ 的超額報酬。
2. 年化使用 $\sqrt{365}$（加密貨幣全年交易），不是常見的 $\sqrt{252}$。
3. $\sqrt{P}$ 年化假設各期獨立。模組 docstring（`sharpe.py:7-13`）特別說明：
   用小時報酬年化會有系統性偏差，同一個 KD 策略用小時報酬算是 −8.70，用日報酬算是 −7.57。
   因此所有 Sharpe 都用**日報酬**計算。`kd_inference.md` 中的「Engine-reported Sharpe (hourly)」
   是引擎從小時資料算出的 −8.70，只作為對照。

### 2.3 Stationary bootstrap 信賴區間（`sharpe.py:84-146`）

- **方法**：Politis 與 Romano（1994）的 stationary bootstrap，由
  `arch.bootstrap.StationaryBootstrap(block_size, values, seed=np.random.default_rng(seed))` 實作。
  區塊長度服從平均值為 $b$ 的幾何分配。
- **block length**：`optimal_block_length(values)["stationary"].iloc[0]`。這是 Politis 與 White（2004）
  的自動選擇法，加上 Patton、Politis 與 White（2009）的修正。估計值會**無條件進位成整數且至少為 1**：
  $b = \max(1, \lceil \hat b \rceil)$；如果估計值不是有限數，則 $b=1$。
- **樣本數門檻**：$n < 20$（`MIN_BOOTSTRAP_OBSERVATIONS`）時回傳 `None`。
- **reps 與 seed**：`kd_inference.md` 使用 `reps = 1000`、`seed = 0`
  （`scripts/demo_statistics.py:69-70`）。
- **每次重抽樣的統計量**：$\widehat{SR}^{*}_{\text{ann}} = \widehat{SR}^{*}_{\text{period}}\cdot\sqrt{365}$。
  未定義（例如 std 為 0）的重抽樣會被丟棄，並記錄成 `valid_reps`。
- **區間**：**百分位數法（percentile）**，$\alpha=0.05$：

$$
\left[\, Q_{\alpha/2}\big(\{\widehat{SR}^{*}\}\big),\ Q_{1-\alpha/2}\big(\{\widehat{SR}^{*}\}\big) \right]
$$

  實作使用 `np.quantile` 的預設線性內插。

⚠ **差異**：這是單純的百分位數區間，沒有偏誤校正，不是 BCa、studentized 或 basic bootstrap。
block length 也被**進位**成整數，而不是直接使用連續的估計值。

本資料的結果：$b = 2$ 天，CI 為 $[-9.18, -5.87]$（見 `kd_inference.md`）。

### 2.4 PSR（`sharpe.py:149-186`）

Bailey 與 López de Prado（2012）：

$$
\mathrm{PSR} = \Phi\!\left(
\frac{(\widehat{SR} - SR^{*})\sqrt{n-1}}
{\sqrt{1 - \hat\gamma_3\,\widehat{SR} + \frac{\hat\gamma_4 - 1}{4}\,\widehat{SR}^2}}
\right)
$$

- $\widehat{SR}$ 是**每期（日）、未年化**的 Sharpe，分母使用 `ddof=1`。$SR^{*}$ = `benchmark_sr` = 0。
- 偏態與峰態都使用**母體動差（除以 $n$）**：

$$
m_k = \frac{1}{n}\sum_d (r_d-\bar r)^k,\quad
\hat\gamma_3 = \frac{m_3}{m_2^{3/2}},\quad
\hat\gamma_4 = \frac{m_4}{m_2^{2}}\ \text{（非超額峰態，常態分配為 3）}
$$

- 根號內的值 $\le 0$ 時回傳 `None`。$\Phi$ 為標準常態分配的 CDF（`statistics.NormalDist`）。

⚠ **差異**：同一個公式裡混用了兩種估計方式：Sharpe 的標準差用樣本估計（`ddof=1`），
偏態峰態用母體動差，而且沒有做小樣本偏誤校正（例如 `scipy.stats.skew(bias=False)` 那種）。
本資料 PSR 為 0.0000。

### 2.5 日 P&L 的 t 統計量（`scripts/demo_statistics.py:283-285`）

$$
t = \frac{\overline{\text{PnL}}}{s_{\text{PnL}}/\sqrt{n}},\qquad s_{\text{PnL}}\ \text{使用 pandas 預設的}\ \texttt{ddof=1}
$$

這裡使用的是**絕對金額 P&L**，不是報酬率，並且假設各日獨立，沒有做自相關修正。
本資料 $n=366$，$t=-7.33$。

### 2.6 損益兩平勝率（`src/tradingdev/domain/statistics/breakeven.py:91-181`）

- 勝場：`net_pnl > 0`；敗場：`net_pnl < 0`；**`net_pnl == 0` 的交易不算勝也不算敗**。

$$
W = \frac{1}{|\mathcal W|}\sum_{k\in\mathcal W}\text{net\_pnl}_k,\qquad
L = \frac{1}{|\mathcal L|}\sum_{k\in\mathcal L}\big|\text{net\_pnl}_k\big|
$$

$$
p^{*} = \frac{1}{1+W/L} = \frac{L}{W+L},\qquad
\hat p = \frac{|\mathcal W|}{|\mathcal W|+|\mathcal L|},\qquad
\text{edge} = \hat p - p^{*}
$$

- **勝率排除零損益交易**，分母是有分出勝負的交易數。docstring（`breakeven.py:14-17`）指出，
  使用這個分母時，edge > 0 若且唯若所有交易的 `net_pnl` 加總為正。
- $W$ 與 $L$ 都使用 `net_pnl`，也就是已扣除 vectorbt fees、成交價已含 slippage 的金額。
- 本資料沒有零損益交易：412 筆交易 = 121 勝 + 291 敗。

**年化換手率**：

$$
\text{notional} = \sum_k \text{size\_quote}_k\left(1+\frac{\text{exit\_price}_k}{\text{entry\_price}_k}\right),\quad
\text{years} = \frac{\text{end}-\text{start}}{365.25\ \text{天}},\quad
\text{turnover} = \frac{\text{notional}/\text{years}}{\text{capital}}
$$

`start` 與 `end` 分別取第一根和最後一根 K 棒的時間戳，所以 years = 365/365.25；
`capital` 是 `init_cash` = 10,000。本資料：notional ≈ 5,014,736，換手率為 501.8 倍。

### 2.7 成本拆解（`scripts/demo_statistics.py:291-294`）

$$
\widehat{\text{cost}} = V \times 2 \times (\text{fees} + \text{slippage}) = V \times 2 \times (0.0006 + 0.0005)
$$

$$
\text{Net} = E_{\text{last}} - E_0,\qquad \widehat{\text{Gross}} = \text{Net} + \widehat{\text{cost}}
$$

- $V$ 是 `backtest_result.metrics["total_volume"]`。舊 pickle 中它的值是
  **只計進場名目金額**的 2,507,242.31（見第一節）。
- 「× 2」把出場那一邊的名目金額**近似為等於進場名目金額**。
- 報告中其他比率：Cost / |Net| 與 Cost / init_cash。

⚠ **差異與注意事項**：

1. 這是**估計值**。交易紀錄的 `fee` 欄位是 0，因為合併前引擎把它寫死為 0，
   實際成本已經含在 `net_pnl` 與成交價裡，無法直接讀出。
2. 出場名目金額實際上是 $\text{size}\times\text{exit\_price}$，不等於進場名目金額，「× 2」只是近似。
3. vectorbt 的 slippage 是調整成交價，不是另外收取的費用。這裡把它視為按名目金額收取的費率。
4. **合併後的語意已經改變**：上游 `src/tradingdev/domain/performance/catalog.py:213-218` 把
   `total_volume` 定義為「進場與出場實際名目金額的總和」。如果在新引擎產生的結果上套用同一個
   公式，出場那一邊會被**重複計算**。

### 2.8 最大回撤（`kd_equity.png`，`scripts/demo_statistics.py:363-408`）

$$
DD_t = \frac{E_t}{\max_{s\le t} E_s} - 1,\qquad \text{Max DD} = \min_t DD_t
$$

這是在**小時頻率**的完整權益曲線上計算的，不是日頻。本資料 Max DD = −64.4%，
與引擎回報的 `max_drawdown` −0.6438 一致。標題中的「10,000 -> 3,612」是 $E_0$ 與 $E_{\text{last}}$，
虛線是 $E_0$。

### 2.9 平原分數（`src/tradingdev/domain/statistics/plateau.py`）

**格點索引**（`plateau.py:393-434`）：把每個參數軸上不重複的值排序後，以排名 $r_k \in \{0,\dots,4\}$
取代原值。之後所有距離都在排名空間中計算，不是用原始參數值。

**鄰域**（`plateau.py:197-201`）：

$$
\mathcal N(\theta) = \left\{\theta' \ne \theta \;:\; \max_k |r_k(\theta') - r_k(\theta)| \le R \right\},\qquad R = 1
$$

- 使用 **Chebyshev 距離**，所以對角線上的格子也算鄰居。半徑使用預設值 `DEFAULT_RADIUS = 1`。
- **排除自身**（`mask[index] = False`）。
- **邊界與角落**：只納入實際存在的格點，不做補值或鏡射，平均的分母就是實際鄰居數。
  在 $5\times5$ 的完整網格中，內部格子有 8 個鄰居，邊上（非角落）有 5 個，角落有 3 個。

**平原分數**（`plateau.py:367-381`）是鄰居 metric 的**等權重算術平均**：

$$
S(\theta) = \frac{1}{|\mathcal N(\theta)|}\sum_{\theta'\in\mathcal N(\theta)} m(\theta')
$$

**最佳點與平原最佳點**（`plateau.py:252-254, 384-390`）：

- best $= \arg\max_\theta m(\theta)$
- plateau_best $= \arg\max_\theta S(\theta)$，沒有鄰居的點不參與比較
- 平手時選輸入順序中較早的那一個

### 2.10 metric_gap 與 grid_distance（`plateau.py:263-287`）

$$
\text{metric\_gap} = m(\text{best}) - m(\text{plateau\_best})
$$

**使用的是 plateau_best 自身的 metric（`plateau_best_metric`），不是它的平原分數 $S$**。
表格中另外列出的 `plateau_best_score` 才是 $S(\text{plateau\_best})$。

$$
\text{grid\_distance} = \max_k |r_k(\text{best}) - r_k(\text{plateau\_best})|
$$

以案例 B 為例：best 是 (20, 0.04)，排名 (1, 3)；plateau_best 是 (10, 0.05)，排名 (0, 4)。
兩者的 Chebyshev 距離是 1，因為 (10, 0.05) 是尖峰的對角鄰居，它的鄰域平均被尖峰本身拉高了。

### 2.11 spikiness（`plateau.py:255-262`）

$$
\text{spikiness} = \frac{m(\text{best}) - \bar m_{\mathcal N}}{s_{\mathcal N}},\qquad
\bar m_{\mathcal N} = \text{best\_neighborhood\_mean},\quad
s_{\mathcal N} = \text{std}\big(\{m(\theta')\}_{\theta'\in\mathcal N(\text{best})},\ \texttt{ddof=1}\big)
$$

- 分子：最佳點的 metric 減去其鄰居的平均。
- 分母：鄰居 metric 的**樣本標準差，`ddof=1`**，不包含最佳點自己。
- 鄰居少於 2 個，或 $s_{\mathcal N}=0$ 時回傳 `None`。
- docstring 把它定位為**次要指標**：鄰居數量少時，分母不穩定。

### 2.12 forest 圖上排：各組參數的 Sharpe 區間（`comparison.py:593-630, 714-778`）

1. 每組參數先用 `period_returns(..., "daily")` 算出日報酬（`scripts/demo_statistics.py:424-429`）。
2. `align_returns` 以 `pd.concat(join="inner")` 取三組參數**共有的日期**。
   圖標題中的「366 common days」就是 `overlap`。
3. **三組資料一起重抽樣**：用同一個 `StationaryBootstrap(block_size, *columns)`，
   每次重抽樣對三組序列使用**相同的日期索引**。
4. block size：分別對**每組序列**計算 Politis–White 估計值，取**最大值**後無條件進位，且至少為 1
   （`comparison.py:788-797`）。全部報酬都相同的序列不參與估計。
5. `reps = 2000`（`COMPARISON_REPS`）、`seed = 0`，統計量為 `partial(sharpe_ratio, periods_per_year=365.0)`。
6. 區間使用百分位數法：$[Q_{0.025}, Q_{0.975}]$。

所以 forest 圖中 KD(14, 3, 3) 的區間 $[-9.20, -5.82]$ 與 `kd_inference.md` 的 $[-9.18, -5.87]$
**不同**，原因是兩者的 reps（2000 對 1000）與 block size 不同：forest 圖取三組中的最大值，
`kd_inference.md` 只用單一序列的估計值。點估計 −7.57 則相同。

### 2.13 forest 圖下排：配對差異（`comparison.py:633-711`）

**同一組索引重抽樣**：先用 `align_returns({"a": a, "b": b})` 對齊日期，再用
`StationaryBootstrap(block_size, values_a, values_b, seed=...)` 重抽樣。
arch 對所有位置參數使用**同一組區塊索引**，所以每次重抽樣中，$a$ 和 $b$ 取到的是同一批日期，
保留了兩者的相關性。block size 是兩組序列 Politis–White 估計值中的較大者，再無條件進位。

$$
\Delta = \widehat{SR}(a) - \widehat{SR}(b),\qquad
\Delta^{*}_b = \widehat{SR}(a^{*}_b) - \widehat{SR}(b^{*}_b)
$$

區間為 $[Q_{0.025}(\Delta^*), Q_{0.975}(\Delta^*)]$，使用百分位數法，`reps = 2000`、`seed = 0`。

**p 值（雙尾）**：

$$
p = \min\!\Big(1,\ 2\cdot\min\big\{\tfrac{1}{B}\textstyle\sum_b \mathbf 1[\Delta^*_b \le 0],\ \tfrac{1}{B}\sum_b \mathbf 1[\Delta^*_b \ge 0]\big\}\Big)
$$

其中 $B$ 是有效重抽樣數。docstring 說明，這個 p 值是讓百分位數區間剛好排除 0 的最小 $1-\text{confidence}$，
所以 p 值與區間的判斷永遠一致。

⚠ **差異**：教科書上的 bootstrap 檢定通常會先把 bootstrap 分配平移到虛無假設下
（例如 $\Delta^* - \Delta$），再看 $|\Delta|$ 落在哪裡。這裡則是直接看**未平移**的
$\Delta^*$ 分配中 0 落在哪一側的尾部，屬於「反轉百分位數區間」的做法。
另外，等於 0 的抽樣兩側都會被計入。

**rho**（`comparison.py:800-806`）：

$$
\rho = \mathrm{corr}_{\text{Pearson}}\big(r^{a}_d,\ r^{b}_d\big),\quad d \in \text{共同日期}
$$

這是**兩組參數各自對齊後、原始（未重抽樣）日報酬**的 Pearson 相關係數（`np.corrcoef`），
不是重抽樣差異的相關係數，也不是權益曲線的相關係數。任一序列全部相同時為 `None`。

**顏色**（`scripts/demo_statistics.py:467`）：區間不包含 0 時用黑色（`INK`），包含 0 時用灰色（`MUTED`）。
圖中三組配對差異的區間都包含 0，所以都是灰色。

---

## 三、真實與合成資料對照

| 圖／檔案 | 部分 | 資料性質 | 來源 |
| --- | --- | --- | --- |
| `plateau_demo.png` | 左圖：案例 A 熱圖 | **合成** | `build_cases`，seed 0，$\mathcal N(0, 0.05^2)$ |
| `plateau_demo.png` | 右圖：案例 B 熱圖 | **合成** | 同上，$\mathcal N(0, 0.3^2)$，加上固定尖峰 2.0 |
| `plateau_demo.png` | 標題的 metric_gap、grid_distance，白圈、黑框 | **合成**資料經真實函式計算 | `plateau_summary` |
| `plateau_demo.png` | 參數軸 lookback、threshold | **虛構**（不對應任何策略） | `LOOKBACKS`、`THRESHOLDS` 常數 |
| `plateau_table.md` | 全部 | **合成** | 同上 |
| `kd_equity.png` | 權益曲線、標題起訖權益 | **真實回測** | `cd03bf19459b48af.pkl` |
| `kd_equity.png` | 回撤面積、Max DD | **真實回測**（由 pickle 推導） | 同上，見 2.8 |
| `comparison_forest.png` | 上排三組 Sharpe 點估計與 CI | **真實回測**（重新執行）加 bootstrap | `run_kd_variants`，seed 0 |
| `comparison_forest.png` | 下排配對差異、p、rho | **真實回測**（重新執行）加 bootstrap | 同上 |
| `comparison_forest.png` | 「366 common days」 | **真實回測** | `align_returns` 的 overlap |
| `kd_inference.md` | 日月數、Sharpe、CI、PSR、t、勝率、換手率 | **真實回測** | `cd03bf19459b48af.pkl` |
| `kd_inference.md` | 成本拆解 | 真實的 `total_volume` 乘上**假設的**費率 | 見 2.7，屬於估計 |

---

## 四、已知限制

以下內容整理自程式碼的 docstring，並補上本次檢查實際發現的問題。

1. **樣本內與樣本外**：`comparison.py:469-471` 指出，比較時只能使用樣本外報酬，
   例如 walk-forward 的測試區間。用調校過參數的樣本內報酬做比較，
   量到的是「哪個策略過度擬合得更徹底」，而不是「哪個策略比較好」。
   本次三組參數都是 2024 全年的**樣本內**報酬，圖的註腳也有標明。
2. **三組參數未經調校**：forest 圖註腳寫的是「untuned parameter sets」。
   KD(9, 3, 3)、KD(14, 3, 3)、KD(21, 5, 5) 是手動挑選的常見組合，沒有做過最佳化，
   所以不能把這張圖當成「最佳參數」的證據。
3. **CI 重疊不代表不顯著**：`comparison.py:465-468, 728-732` 說明，上排各自的區間忽略了
   不同策略之間的相關性，所以只是保守的初步篩選。區間不重疊可以推論差異顯著，
   但區間重疊**不能**推論不顯著，正式的判斷要看下排的配對差異。
4. **統計顯著不等於經濟顯著**：`comparison.py:475-477` 指出，估計很精確但很小的優勢，
   可能撐不過交易成本或容量限制。
5. **比較的期間、標的與成本必須一致**：`comparison.py:472-474` 指出，這是實驗設計的問題，
   統計方法無法修補；程式只保證取共同期間。
6. **`cost_per_turnover` 為何低估**：`breakeven.py:21-23` 指出，成本是交易紀錄 `fee` 欄位的加總。
   如果引擎把成本算進 `net_pnl`，但沒有寫進 `fee`，這個數字就會低估。
   合併前的引擎把 `fee` 寫死為 0.0，fees 由 vectorbt 從現金扣除，slippage 含在成交價中，
   所以這份 pickle 的 `total_cost = 0`、`cost_per_turnover = 0`，**完全看不到成本**。
   因此 `kd_inference.md` 改用 2.7 的外部估計。
   合併後的新引擎（`src/tradingdev/domain/backtest/metrics.py:49`）會把 `entry_fees + exit_fees`
   寫入 `fee`，但 catalog 說明 signal 模式的 slippage 仍然「embedded in fill prices」，
   所以 slippage 那一部分仍然不會出現在 `fee` 中。
7. **平原分析不能取代樣本外驗證**：`plateau.py:53-56` 指出，如果整個參數區域都對同一個市場
   型態過度擬合，所有鄰居會因為同樣的原因看起來都很好，形成假平原。
8. **平原分析的設計取捨**：`plateau.py:22-40` 指出：
   - 網格不均勻時，半徑 1 對應的實際參數變化量並不固定。
   - 參數超過 3 個時，鄰域最多有 $3^d-1$ 個點，已經不算局部，會以 `dimension_warning` 標示。
   - spikiness 的分母是少數鄰居的樣本標準差，數值不穩定。
9. **小時與日頻 Sharpe**：`sharpe.py:7-13` 指出，小時報酬有自相關，用 $\sqrt{P}$ 年化會產生偏差。
10. **2024-12-31 只有一根 K 棒**：最後一天的日報酬只涵蓋 1 小時，但仍算成 1 天，見 2.1。
11. **資料來源是推定的**：pickle 沒有記錄 `source` 與 `market_type`，見第一節。
12. **`total_volume` 的定義在合併前後不同**：見 2.7 第 4 點。

---

## 五、可重現性

### 固定的 seed 與常數

| 用途 | 值 | 位置 |
| --- | --- | --- |
| 平原合成資料 | `np.random.default_rng(0)` | `scripts/demo_plateau.py:35, 54`（`demo_statistics.py:69, 90` 相同） |
| `kd_inference.md` 的 Sharpe CI | `seed=0`、`reps=1000` | `scripts/demo_statistics.py:69-70, 276-281` |
| forest 圖的 `interval_table` | `seed=0`、`reps=2000` | `scripts/demo_statistics.py:66, 431` |
| forest 圖的 `difference_interval` | `seed=0`、`reps=2000`（每一對都重新以 seed 0 開始） | `scripts/demo_statistics.py:434-439` |
| 年化期數 | 365.0 | `scripts/demo_statistics.py:68` |
| 成本費率 | fees 0.0006、slippage 0.0005 | `scripts/demo_statistics.py:71-72` |
| 平原半徑 | 1 | `plateau.py:74` |

### 重現指令

所有指令都在 repository 根目錄執行，並先完成 `uv sync --locked --all-extras`。

**`plateau_demo.png` 與 `plateau_table.md`**

```bash
uv run python scripts/demo_plateau.py
```

完全使用合成資料，不需要網路或快取。連續執行兩次，`plateau_table.md` 的 md5 相同
（`5e5ad3d8ba99fb03dd3df1e64c798f95`）。

**`kd_equity.png`、`comparison_forest.png` 與 `kd_inference.md`**

```bash
uv run python scripts/demo_statistics.py
```

注意事項：

1. 這支腳本**目前沒有納入 git**，檔案遺失就無法重現。
2. 它的 `main()` **也會寫入 `plateau_demo.png` 與 `plateau_table.md`**，用的是軸標籤為網格索引的
   舊版，會覆蓋 `demo_plateau.py` 的輸出。若兩者都要，請**先**執行 `demo_statistics.py`，
   **再**執行 `demo_plateau.py`。
3. `kd_equity.png` 與 `kd_inference.md` 讀取的是 mtime 最新的 pickle。如果之後產生了更新的
   pickle，結果就會改變。若要固定輸入，必須確認 `cd03bf19459b48af.pkl` 仍是最新的那一個。
4. `comparison_forest.png` 會用**目前的程式碼重新執行三次回測**，需要能取得 BTC/USDT 1h
   的資料，可以讀本機快取或從 binance_vision 下載。現有這張圖是合併前的引擎產生的；
   合併後重跑，回測結果與 Sharpe 可能不同。這點本文件沒有驗證。
5. 本次合併後已實際重跑過 `write_kd_inference`，輸出的 `kd_inference.md` 與合併前那份
   **逐字相同**。至於 `kd_equity.png` 與 `comparison_forest.png`，合併後沒有重新產生。
