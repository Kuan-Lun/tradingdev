# 展示圖方法說明（METHODS）

本文件說明 `demo_output/` 中三張圖（`plateau_demo.png`、`kd_equity.png`、
`comparison_forest.png`）每個數字的資料來源、公式與程式碼位置；
附帶的 `plateau_table.md`、`kd_inference.md` 也一併涵蓋。

內容全部取自以下來源的實際讀取或執行結果，沒有依描述推測：

- `scripts/demo_plateau.py`（commit `fcebb68`）
- `scripts/demo_statistics.py`（commit `7bbcdce` 納入 git，之後隨分支更新）
- `src/tradingdev/domain/statistics/{periods,performance,sharpe,breakeven,plateau,comparison}.py`
- `workspace/data/processed/cache/cd03bf19459b48af.pkl`（以 `pickle.load` 實際讀取）
- `src/tradingdev/domain/strategies/bundled/kd_strategy/config.yaml`
- 合併前（`6826e41`）的 `src/tradingdev/domain/backtest/{signal_engine,metrics,base_engine}.py`
- 第三方實作：`arch` 8.0.0（`arch/bootstrap/base.py`、`_samplers_python.py`）、`vectorbt` 1.0.0
  （分析合併前引擎時，合併前的 `uv.lock` 鎖定的版本；目前鎖定的版本見第五節）

實作與教科書寫法不同之處，以 **⚠ 差異** 標註。文中的「實測」表示本次實際執行程式得到的值。

---

## 一、資料來源

### 1. `plateau_demo.png`：合成資料

目前 `demo_output/plateau_demo.png` 是 `scripts/demo_plateau.py` 產生的版本
（viridis 色階、軸標籤為 lookback／threshold；檔案 mtime 2026-10-11 14:51:40 UTC，
與 commit `fcebb68` 同時）。資料完全不經過回測。

**產生函式**：`build_cases()`，`scripts/demo_plateau.py:52-67`。

**參數軸的實際值**（`scripts/demo_plateau.py:36-37`）：

| 軸 | 迴圈索引 | 參數值 |
| --- | --- | --- |
| x：`lookback` | $i = 0,\dots,4$ | 10, 20, 30, 40, 50（以 `float(lookback)` 存入 `Trial.params`） |
| y：`threshold` | $j = 0,\dots,4$ | 0.01, 0.02, 0.03, 0.04, 0.05 |

這兩個參數軸是虛構的，不對應任何實際策略。

**亂數種子與分布**（`scripts/demo_plateau.py:35, 54-57`）：
`rng = np.random.default_rng(SEED)`，`SEED = 0`。依下列固定順序抽兩個 $5\times5$ 矩陣：

1. $\varepsilon^{A}_{ij} \sim \mathcal N(0,\,0.05^2)$：`rng.normal(0.0, 0.05, (5, 5))`
2. $\varepsilon^{B}_{ij} \sim \mathcal N(0,\,0.3^2)$：`rng.normal(0.0, 0.3, (5, 5))`

`numpy` 的 `normal(loc, scale, size)` 中，`scale` 是標準差。

**案例 A（真平原，`scripts/demo_plateau.py:59-63`）**：

$$
m^{A}_{ij} = 1.5 - 0.3\,|i-2| - 0.3\,|j-2| + \varepsilon^{A}_{ij}
$$

**案例 B（假尖峰，`scripts/demo_plateau.py:64-66`）**：

$$
m^{B}_{ij} =
\begin{cases}
2.0 & (i,j) = (1,3)\ \text{，即 lookback}=20,\ \text{threshold}=0.04 \\
\varepsilon^{B}_{ij} & \text{其他格子}
\end{cases}
$$

$\varepsilon^{B}_{1,3}$ 仍然會被抽出，只是被 2.0 覆蓋，所以不影響其他格子的亂數。

**繪圖細節**（`scripts/demo_plateau.py:76-186`）：

- 兩張熱圖使用**共同色階**：$v_{\min}$、$v_{\max}$ 是兩個案例所有格子的最小值與最大值
  （`:89-90`）。實際值為 $-0.38$ 與 $2.00$。
- 格內數字是 `plateau_grid(...).metric` 的值，取小數點後 2 位（`:97-107`）。
- 白色圓圈標示 `best_params`，黑色方框標示 `plateau_best_params`（`:108-128`）。
- 子圖標題的 `metric_gap`、`grid_distance` 來自 `plateau_summary`（`:129-134`）。

> 注意：`scripts/demo_statistics.py` 的 `plateau_cases()` 與 `plot_plateau()` 有另一份平原繪圖程式，亂數順序與公式相同，
> 但 `Trial.params` 直接使用網格索引 `{"x": i, "y": j}`，色階為 `Blues`。它的 `main()`
> 寫入另外的 `plateau_gridindex.png` 與 `plateau_gridindex_table.md`，不會覆蓋 `demo_plateau.py` 的輸出。
> 由於鄰域只看排名（見 2.9），兩份程式算出的統計數字相同，只有軸標籤與配色不同。

### 2. `kd_equity.png`：真實回測（讀取快取 pickle）

`latest_cache()`（`scripts/demo_statistics.py`）依**檔案修改時間**挑選
`workspace/data/processed/cache/*.pkl` 中最新的一個；它不檢查 pickle 的內容是不是 KD 策略。

快取目錄目前有兩個檔案：`cd03bf19459b48af.pkl`（mtime 2026-10-05 06:33:29 UTC）與
`e9d1e13c2eb25909.pkl`（mtime 2026-09-24 09:21:07 UTC）。本次以 `cmp` 確認兩者**逐位元組相同**，
腳本選中的是前者。

以下各值都是從 pickle 實際讀出：

| 項目 | 值 | 讀取位置 |
| --- | --- | --- |
| pickle 檔名 | `workspace/data/processed/cache/cd03bf19459b48af.pkl` | 檔案系統 |
| 物件型別 | `PipelineResult`，`mode = "simple"`，`fold_results` 為空 | pickle 頂層 |
| 策略 id | `kd_crossover`（version `1.0.0`） | `config_snapshot.strategy` |
| 策略類別 | `KDStrategy`，`source_path = src/tradingdev/domain/strategies/bundled/kd_strategy/strategy.py` | 同上 |
| 參數組合 | `k_period=14, d_period=3, smooth_k=3, overbought=80.0, oversold=20.0` | `config_snapshot.strategy.parameters` |
| 交易對 | `BTC/USDT` | `config_snapshot.backtest.symbol` |
| market_type | **pickle 未記錄**；程式預設為 `futures/um`（見下方說明） | — |
| 時間框架 | `1h` | `config_snapshot.backtest.timeframe` |
| 起訖日期（config） | `2024-01-01` 至 `2024-12-31` | `config_snapshot.backtest.start_date/end_date` |
| 實際第一根／最後一根 K 棒 | `2024-01-01 00:00:00+00:00` / `2024-12-31 00:00:00+00:00` | `backtest_result.timestamps` |
| 資料來源 crawler | **pickle 未記錄**；程式預設為 `binance_vision`（見下方說明） | — |
| fees | `0.0006` | `config_snapshot.backtest.fees` |
| slippage | `0.0005` | `config_snapshot.backtest.slippage` |
| init_cash | `10000.0` | `config_snapshot.backtest.init_cash`、`backtest_result.init_cash` |
| K 棒根數 | 8,761（相鄰時間戳的間隔只有 3600 秒一種，沒有缺口） | `len(backtest_result.equity_curve)` |
| 交易筆數 | 412（121 勝、291 敗、0 筆零損益） | `len(backtest_result.trades)` |
| 起訖權益 | 10,000.00 → 3,611.66 | `equity_curve[0]`、`equity_curve[-1]` |
| 回測模式 | `signal` | `backtest_result.mode` |

**資料來源與 market_type**：pickle 的 `config_snapshot.data.requirements.market` 只有
`{symbol, timeframe}`，沒有 `source` 或 `market_type`。目前程式的預設值是
`source = "binance_vision"`（`src/tradingdev/domain/data/requirements.py:13`、
`src/tradingdev/domain/data/schemas.py:50`、`src/tradingdev/app/data_service.py:76`）與
`market_type = "futures/um"`（`schemas.py:51`）。因此**推定**使用這兩個預設值，但 pickle 本身無法證明。
本機資料檔為 `workspace/data/raw/btcusdt_1h_2024.csv` 與 `workspace/data/processed/btcusdt_1h_2024.parquet`。

**產生 pickle 的引擎**：pickle 本身沒有記錄 commit。它的欄位特徵與合併前 `6826e41` 的
signal engine 一致：

- `vbt.Portfolio.from_signals(..., price=open_, fees=0.0006, slippage=0.0005, freq="1h")`
  （`6826e41:signal_engine.py:55-79`，`freq` 預設值見 `6826e41:base_engine.py:26`）。
  訊號先 `shift(1)`，以下一根 K 棒的 open 成交。
- `_extract_trades` 把每筆交易的 `fee` 寫死為 `0.0`，並令 `gross_pnl = net_pnl = PnL`
  （`6826e41:signal_engine.py:122-125`）。實測 412 筆交易的 `fee` 加總為 0。
- `metrics["total_volume"]` $=\sum|\text{Size}|\times\text{Avg Entry Price}$，**只計進場名目金額**
  （`6826e41:metrics.py:25-32`）。實測為 2,507,242.31，正好等於 $\sum$ `size_quote`。

### 3. `comparison_forest.png`：真實回測（重新執行，不讀 pickle）

`run_kd_variants()`（`scripts/demo_statistics.py`）用 `load_config` 讀入
`src/tradingdev/domain/strategies/bundled/kd_strategy/config.yaml`，深拷貝後只覆蓋三組參數，
再呼叫 `BacktestService().run_raw_config(config)` 重新回測。

config 的共用設定：策略 `kd_crossover`／`KDStrategy`、`BTC/USDT`、`1h`、`2024-01-01` 至 `2024-12-31`、
`init_cash = 10000.0`、`fees = 0.0006`、`slippage = 0.0005`、`periods_per_year = 365.0`。
config 沒有指定 `source` 與 `market_type`，所以適用上述預設值 `binance_vision`／`futures/um`。

| 標籤 | k_period | d_period | smooth_k | overbought / oversold | K 棒根數 | 交易筆數（勝／敗／零） | 期末權益 |
| --- | ---: | ---: | ---: | --- | ---: | --- | ---: |
| KD(9, 3, 3) | 9 | 3 | 3 | 80.0 / 20.0（沿用 config） | 8,761 | 367（112／255／0） | 4,274.23 |
| KD(14, 3, 3) | 14 | 3 | 3 | 80.0 / 20.0 | 8,761 | 412（121／291／0） | 3,611.66 |
| KD(21, 5, 5) | 21 | 5 | 5 | 80.0 / 20.0 | 8,761 | 267（77／190／0） | 5,509.81 |

**K 棒根數與交易筆數的取得方式**：腳本執行時沒有保存這些數字。本表是本次用**目前程式碼**
（合併後）重跑同樣三組設定實測所得。重跑後的 bootstrap 結果與圖上**所有標註數字完全一致**
（三個點估計與 CI、三個差異的點估計、CI、p、rho、366 common days），
所以日報酬序列與產生該圖時相同，本表可以代表圖中所用的回測。

圖檔 mtime 為 2026-10-11 14:43:17 UTC，早於合併 commit `b4aae8d`（14:49:07 UTC），
所以圖是合併前的引擎產生的。合併前後引擎的差異只影響 `fee` 與 `total_volume` 欄位（見 2.7），
不影響權益曲線。KD(14, 3, 3) 重跑後的期末權益與 pickle 完全相同。

---

## 二、計算公式

符號：$E_t$ 為第 $t$ 根 K 棒收盤時的權益，$E_0$ 為權益曲線的第一個值，$\text{ts}_t$ 為其時間戳。

### 2.1 日頻報酬（`src/tradingdev/domain/statistics/periods.py`）

**UTC 界線**（docstring `periods.py:3-13`；`to_utc_index`、`_parse_each`，`periods.py:36-79`）：

- 一天是半開區間 $[00{:}00{:}00,\ 24{:}00{:}00)$ UTC；一個月是當月 1 日 00:00 UTC 到下月 1 日 00:00 UTC。
- 時間戳**依原標籤**歸入所屬期間。Binance 以開盤時間標示 K 棒，所以 23:00 那根 K 棒算在它開盤的那一天。
- 無時區的時間戳視為 UTC，有時區的先轉為 UTC。字串與物件逐一解析、逐一轉換，所以 offset 不同的時間戳（例如日光節約時間切換前後的當地時間）也能使用；同一序列中無時區與有時區混用則 raise `ValueError`。
- `equity_series`（`periods.py:82-106`）會檢查：權益曲線非空、全為有限值、長度與時間戳相同、
  時間戳嚴格遞增。

**期末權益**（`period_closing_equity`，`periods.py:109-122`）：
`series.resample("D").last().dropna()`，月頻則用 `"MS"`。

$$
C_d = E_{t^*(d)},\qquad t^*(d) = \max\{t : \text{ts}_t \in d\}
$$

**第一期的基準**（`_opening_equity`，`periods.py:163-165`）：
`closing.shift(1).fillna(E_0)`。第一期的期初權益是權益曲線的第一個值 $E_0$，
不是第一天之前某個收盤值。

$$
O_{d_1} = E_0,\qquad O_{d_k} = C_{d_{k-1}}\ \ (k \ge 2)
$$

**報酬與損益**（`period_pnl`，`periods.py:125-134`；`period_returns`，`periods.py:137-155`）：

$$
\text{PnL}_d = C_d - O_d,\qquad r_d = \frac{C_d}{O_d} - 1
$$

因此 $\sum_d \text{PnL}_d = E_{\text{last}} - E_0$，$\prod_d (1+r_d) = E_{\text{last}}/E_0$。
只要有任何 $E_t \le 0$，`period_returns` 就會 raise `ValueError`（`periods.py:151-152`）。

**缺資料期間**：沒有觀測值的期間由 `.dropna()` **直接略過，不補 0**（docstring `periods.py:17-19`）。
⚠ **差異**：若中間有缺口，缺口後第一期的報酬會涵蓋整段缺口，並且只算 1 期；
常見做法是把缺口日記為 0 報酬，或保留 NaN。本資料沒有缺口，不受影響。

**本資料實測**：日數 366、月數 12。最後一根 K 棒是 `2024-12-31 00:00`，所以 2024-12-31
**只有 1 根 K 棒**，這一天的「日報酬」其實只是 1 小時的權益變化。

`kd_inference.md` 的「Share of months with negative P&L」
$=\frac{1}{M}\sum_m \mathbf 1[\text{PnL}_m<0]$（`scripts/demo_statistics.py` 的 `write_kd_inference()`），
實測為 12/12。

### 2.2 年化 Sharpe（`sharpe.py:144-156`、`_period_sharpe`，`sharpe.py:448-453`）

$$
\widehat{SR}_{\text{day}} = \frac{\bar r}{s},\quad
s = \sqrt{\frac{1}{n-1}\sum_{d=1}^{n}(r_d-\bar r)^2},\qquad
\widehat{SR}_{\text{ann}} = \widehat{SR}_{\text{day}}\cdot\sqrt{P}
$$

- `periods_per_year` $P$ 的實際值是 **365.0**，來自 `DAILY_PERIODS_PER_YEAR`
  （`scripts/demo_statistics.py` 的常數 `DAILY_PERIODS_PER_YEAR`），在 `write_kd_inference()` 與
  `plot_comparison_forest()` 傳入。
- 標準差使用 `ddof=1`，無風險利率為 0（docstring `sharpe.py:17-19`）。
- $n<2$，或 `np.ptp(values) == 0`（所有報酬完全相等）時回傳 `None`。

⚠ **差異**：

1. 沒有扣除無風險利率。
2. 年化因子是 $\sqrt{365}$（加密貨幣全年交易），不是股市常用的 $\sqrt{252}$。
3. $\sqrt{P}$ 的縮放假設各期獨立。docstring（`sharpe.py:7-13`）說明小時報酬有自相關，
   用小時報酬年化會有系統性偏差（同一個 KD 策略：小時 −8.70、日 −7.57），所以一律用日報酬。

`kd_inference.md` 的「Engine-reported Sharpe (hourly)」是 pickle 中的 `metrics["sharpe_ratio"]`
（−8.70，`scripts/demo_statistics.py` 的 `write_kd_inference()`）。它是合併前引擎呼叫 vectorbt 的 `pf.sharpe_ratio()` 得出的值
（`6826e41:metrics.py:51`），使用逐根 K 棒報酬，`freq="1h"`，vectorbt 的 `year_freq` 為 365 天，
所以年化因子是 $\sqrt{8760}$。本文件只把它當作對照，圖中沒有使用。

### 2.3 Stationary bootstrap 信賴區間（`sharpe_interval_outcome`，`sharpe.py:182-259`）

`sharpe_confidence_interval`（`sharpe.py:159-179`）呼叫 `sharpe_interval_outcome`，並把所有 `IntervalUnavailable` 轉成 `None`。

- **方法**：Politis 與 Romano（1994）的 stationary bootstrap，用
  `arch.bootstrap.StationaryBootstrap(block_size, values, seed=np.random.default_rng(seed))` 實作
  （`sharpe.py:242-244`）。
  每個位置以機率 $1/b$ 開始新區塊（起點均勻抽取），否則延續前一個索引 +1，
  所以區塊長度服從平均 $b$ 的幾何分配。
  ⚠ arch 的實作會**循環接回**：索引超過 $n-1$ 時回到 0（`arch/bootstrap/_samplers_python.py:29-33`）。
- **block length 的估計**（`sharpe.py:240-241`）：
  呼叫 `arch.bootstrap.optimal_block_length(values)`，取回傳表格 `stationary` 欄的第一列。
  它採用 Politis 與 White（2004）〈Automatic Block-Length Selection for the Dependent Bootstrap〉，
  並加上 Patton、Politis 與 White（2009）的更正（docstring `sharpe.py:198-206`；
  arch 文獻引用見 `arch/bootstrap/base.py:185-190`）。

$$
b = \begin{cases}\max\big(1,\ \lceil \hat b_{\text{PW}} \rceil\big) & \hat b_{\text{PW}}\ \text{為有限值}\\ 1 & \text{否則}\end{cases}
$$

⚠ **差異**：估計值被**無條件進位成整數**，而不是直接使用連續值。

- **樣本數門檻**：$n < 20$（`MIN_BOOTSTRAP_OBSERVATIONS`，`sharpe.py:45`）時沒有區間。
  `sharpe_interval_outcome` 回傳 `IntervalUnavailable`（`sharpe.py:53-92`），
  `reason` 為 `"insufficient_observations"`，並附上實際樣本數、門檻與點估計。
  門檻以期數計算，所以月報酬需要 20 個月，一年的月報酬永遠沒有區間。
  Sharpe 本身未定義時 `reason` 為 `"undefined_statistic"`，沒有任何有效重抽樣時為 `"no_valid_resample"`。
- **reps 與 seed**：`kd_inference.md` 使用 `reps = 1000`（`BOOTSTRAP_REPS`）、`seed = 0`（`SEED`），
  見 `scripts/demo_statistics.py` 的常數 `SEED`、`BOOTSTRAP_REPS` 與 `write_kd_inference()`。
- **每次重抽樣的統計量**（`sharpe.py:245-246`）：
  $\widehat{SR}^{*}_{\text{ann}} = \widehat{SR}^{*}_{\text{day}}\cdot\sqrt{365}$。
  未定義的重抽樣（全部報酬相等）會變成 NaN 並被丟棄，剩下的數量記在 `valid_reps`。
- **區間方法：百分位數法**（`sharpe.py:249-250`），信賴水準為預設的 0.95：

$$
\Big[\ Q_{0.025}\big(\{\widehat{SR}^{*}_{\text{ann}}\}\big),\ \ Q_{0.975}\big(\{\widehat{SR}^{*}_{\text{ann}}\}\big)\ \Big]
$$

  $Q$ 使用 `np.quantile` 的預設線性內插。
  ⚠ **差異**：這是單純的 percentile interval，沒有偏誤校正，也不是 BCa、studentized 或 basic bootstrap。

**本資料實測**：$b = 2$ 天，CI 為 $[-9.18, -5.87]$，`valid_reps = 1000`。

### 2.4 PSR（`probabilistic_sharpe_ratio`，`sharpe.py:262-299`）

Bailey 與 López de Prado（2012）：

$$
\mathrm{PSR} = \Phi\!\left(
\frac{(\widehat{SR} - SR^{*})\sqrt{n-1}}
{\sqrt{1 - \hat\gamma_3\,\widehat{SR} + \dfrac{\hat\gamma_4 - 1}{4}\,\widehat{SR}^2}}
\right)
$$

- $\widehat{SR}$ 是**每期（日）、未年化**的 Sharpe，由 `_period_sharpe` 計算，分母是 `ddof=1` 的標準差。
- $SR^{*}$ 是 `benchmark_sr`，腳本使用預設值 0（`scripts/demo_statistics.py` 的 `write_kd_inference()`）。
- **偏態與峰態使用母體動差（除以 $n$）**（`sharpe.py:291-295`）：

$$
m_k = \frac{1}{n}\sum_{d}(r_d-\bar r)^k,\qquad
\hat\gamma_3 = \frac{m_3}{m_2^{3/2}},\qquad
\hat\gamma_4 = \frac{m_4}{m_2^{2}}\ \ \text{（非超額峰態，常態分配為 3）}
$$

- 根號內的值 $\le 0$ 時回傳 `None`（`sharpe.py:296-297`）。$\Phi$ 是 `statistics.NormalDist().cdf`。

⚠ **差異**：同一個式子裡，Sharpe 的標準差用樣本估計（`ddof=1`），偏態與峰態卻用母體動差，
也沒有做小樣本偏誤校正。本資料 PSR = 0.0000。

### 2.5 日 P&L 的 t 統計量（`scripts/demo_statistics.py` 的 `write_kd_inference()`）

$$
t = \frac{\overline{\text{PnL}}_{\text{day}}}{s_{\text{PnL}}/\sqrt{n}}
$$

$s_{\text{PnL}}$ 是 `pandas.Series.std(ddof=1)`。這裡使用絕對金額 P&L 而不是報酬率，並假設各日獨立。
本資料 $n = 366$，$t = -7.33$。

### 2.6 損益兩平勝率（`summarize_breakeven`，`breakeven.py:104-198`）

腳本呼叫方式：`summarize_breakeven(result.trades, start=stamps[0], end=stamps[-1], capital=result.init_cash)`
（`scripts/demo_statistics.py` 的 `write_kd_inference()`）。

**勝敗定義**（`breakeven.py:157-158`）：$\mathcal W = \{k : \text{net\_pnl}_k > 0\}$，
$\mathcal L = \{k : \text{net\_pnl}_k < 0\}$。`net_pnl == 0` 的交易兩邊都不算。

$$
W = \frac{1}{|\mathcal W|}\sum_{k\in\mathcal W}\text{net\_pnl}_k,\qquad
L = \frac{1}{|\mathcal L|}\sum_{k\in\mathcal L}\big|\text{net\_pnl}_k\big|
$$

$W$ 是勝場的平均淨損益，$L$ 是敗場的平均淨虧損絕對值（`breakeven.py:159-163`）。

$$
p^{*} = \frac{1}{1+W/L} = \frac{L}{W+L},\qquad
\hat p = \frac{|\mathcal W|}{|\mathcal W|+|\mathcal L|},\qquad
\text{edge} = \hat p - p^{*}
$$

（`breakeven.py:164-168, 181-185`）

- **勝率排除零損益交易**，分母是分出勝負的交易數（docstring `breakeven.py:13-16`）。
  docstring 指出，用這個分母時，edge > 0 若且唯若所有交易的 `net_pnl` 加總為正。
  `performance.py` 的 `win_rate` 則用全部交易數當分母，兩者定義不同。
- $W$、$L$ 都根據 `net_pnl`。這份 pickle 的 `net_pnl` 是 vectorbt 已扣 fees 的 PnL，
  成交價已含 slippage。
- 本資料沒有零損益交易：412 = 121 + 291。$W/L = 0.60$，$p^* = 62.5\%$，$\hat p = 29.4\%$，
  edge = −33.1%。

**年化換手率**（`breakeven.py:153-154, 170, 189-191, 201-206`）：

$$
\text{notional} = \sum_k \text{size\_quote}_k\left(1+\frac{\text{exit\_price}_k}{\text{entry\_price}_k}\right),\quad
\text{years} = \frac{\text{end}-\text{start}}{365.25\ \text{天}},\quad
\text{turnover} = \frac{\text{notional}/\text{years}}{\text{capital}}
$$

出場名目金額以 $\text{size\_quote}\times\text{exit}/\text{entry}$ 推算。`status` 為 `open` 的未平倉交易只算進場腿，因為它的 `exit_price` 是最後的標記價、不是成交價；它仍留在勝敗樣本中（理由見 docstring）。這份資料 412 筆都已平倉，不受影響。`start` 與 `end` 分別取第一根與最後一根
K 棒的時間戳，因此 years $=365/365.25$。`start` 或 `end` 缺失（`None`、NaN、NaT）時 raise `ValueError`。capital 是 `init_cash = 10000`。
實測 notional = 5,014,735.72，換手率 501.8 倍。

### 2.7 成本拆解（`write_kd_inference`，`scripts/demo_statistics.py`）

$$
\widehat{\text{cost}} = V \times (\text{fees} + \text{slippage}) = V \times (0.0006 + 0.0005)
$$

$$
\text{Net} = E_{\text{last}} - E_0,\qquad
\widehat{\text{Gross}} = \text{Net} + \widehat{\text{cost}}
$$

- $V$ 是 `summarize_breakeven` 回傳的 `traded_notional`，由交易紀錄計算的進場加出場名目金額（見 2.6），
  實測 5,014,735.72。不使用 `metrics["total_volume"]`，因為該欄位的單腿／雙腿慣例在上游改過（見第 4 點）。
- `fees`、`slippage` 是腳本中寫死的常數 `FEES`、`SLIPPAGE`（`scripts/demo_statistics.py`），
  不是從 pickle 讀出（數值剛好與 config 相同）。
- 報表中的其他比率：$\widehat{\text{cost}}/|\text{Net}|$、$\widehat{\text{cost}}/\text{init\_cash}$。
- 實測：cost = 5,516.21，Gross = −872.13，Net = −6,388.34，cost/|Net| = 86.3%，cost/init_cash = 55.2%。

⚠ **差異與注意事項**：

1. 這是**估計值**，不是 `breakeven.py` 的 `total_cost`。後者是交易紀錄 `fee` 欄位的加總，
   而這份 pickle 的 `fee` 全為 0。
2. 出場名目以 $\text{size\_quote}\times\text{exit}/\text{entry}$ 推算（見 2.6），不是假設等於進場名目。
3. vectorbt 的 slippage 是把成交價往不利方向調整，不是另外收取的費用。這裡把它當作按名目金額收取的費率。
4. **合併後 `total_volume` 的語意改變**：`src/tradingdev/domain/performance/catalog.py:213-217`
   定義為「Sum of actual entry and exit notional」。用目前引擎重跑 KD(14, 3, 3)，
   實測 `total_volume` = 5,014,735.72（已含出場）。舊版腳本用 `total_volume × 2` 估計，對新引擎的結果會
   **重複計算出場一側**，所以腳本已改用 `traded_notional`。
   目前引擎會把 `entry_fees + exit_fees` 寫入 `fee`（`src/tradingdev/domain/backtest/metrics.py:49`），
   實測 `fee` 加總為 3,008.84，約等於 $5{,}014{,}735.72 \times 0.0006$；slippage 仍然內含於成交價，
   不會出現在 `fee` 中（`catalog.py:233-234`）。

### 2.8 權益曲線與最大回撤（`plot_kd_equity`，`scripts/demo_statistics.py`）

$$
DD_t = \frac{E_t}{\max_{s\le t} E_s} - 1,\qquad \text{Max DD} = \min_t DD_t
$$

- 回撤在**逐小時**的完整權益曲線上計算，不是日頻。圖中以百分比呈現。
- 實測 Max DD = −64.4%，與 pickle 中引擎的 `max_drawdown` = −0.6438 一致。
- 上圖虛線為 $E_0$；標題中的「10,000 -> 3,612」是 $E_0$ 與 $E_{\text{last}}$。
- 標題文字「KD crossover, BTC/USDT 1h」是**寫死在程式中**的字串，不是從 pickle 讀出；
  它剛好與這份 pickle 的內容相符。

### 2.9 平原分數（`src/tradingdev/domain/statistics/plateau.py`）

**格點索引**（`plateau.py` 的 `_build_grid`）：把呼叫端傳入的 `axes`（每個參數「搜尋過」的值，
包含整排失敗的值）排序，以排名 $r_k \in \{0,\dots,4\}$ 取代原值。之後所有距離都在排名空間中計算，
不使用原始參數值。兩支 demo 的網格都是完整的，所以 `axes` 與 trial 實際用到的值相同。

**鄰域**（`plateau.py` 的 `_Grid.neighbours`；模組 docstring 的 First-version design decisions）：

$$
\mathcal N(\theta) = \Big\{\theta' \ne \theta \;:\; \max_k |r_k(\theta') - r_k(\theta)| \le R \Big\},\qquad R = 1
$$

- 距離使用 **Chebyshev 距離**，所以對角格也算鄰居。
- 半徑使用 `plateau.py` 的常數 `DEFAULT_RADIUS = 1`，腳本沒有另外指定。
- **排除自身**：`_Grid.neighbours` 中的 `mask[index] = False`。模組 docstring 說明這是為了讓孤立尖峰不能靠自己的值得分。
- **邊界與角落**：只納入實際存在的試驗點，不補值、不鏡射、不循環，平均的分母就是實際鄰居數
  （模組 docstring 的 Conventions）。在完整的 $5\times5$ 網格中，內部格有 8 個鄰居，邊上非角落的格子有 5 個，角落格有 3 個。

**平原分數**（`plateau.py` 的 `_scores`）是鄰居 metric 的等權重算術平均：

$$
S(\theta) = \frac{1}{|\mathcal N(\theta)|}\sum_{\theta'\in\mathcal N(\theta)} m(\theta')
$$

**best 與 plateau_best**（`plateau_summary`、`_argmax_score`）：

- best $= \arg\max_\theta m(\theta)$（`np.argmax`，平手時取輸入順序中最早的一個）
- plateau_best $= \arg\max_\theta S(\theta)$，沒有鄰居的點不參與比較；平手時同樣取最早的一個。
  $S$ 是不同鄰居數的平均，相同的 metric 也可能因捨入差幾個 ulp，所以與最高分相差在相對 $10^{-12}$ 以內（以兩個比較分數的絕對值較大者為尺度）即視為平手。metric 本身是輸入值，只有完全相等才算平手。

### 2.10 metric_gap 與 grid_distance（`plateau_summary`）

$$
\text{metric\_gap} = m(\text{best}) - m(\text{plateau\_best})
$$

**使用的是 plateau_best 自身的 metric（`plateau_best_metric`），不是它的平原分數 $S$。**
表格中的 `plateau_best_score` 才是 $S(\text{plateau\_best})$。

$$
\text{grid\_distance} = \max_k |r_k(\text{best}) - r_k(\text{plateau\_best})|
\quad(\texttt{\_Grid.distance})
$$

**實測**：

- 案例 A：best 與 plateau_best 都是 (30, 0.03)，metric_gap = 0.000，grid_distance = 0。
- 案例 B：best 是 (20, 0.04)，m = 2.000；plateau_best 是 (10, 0.05)，m = 0.066，
  所以 metric_gap = 1.934，grid_distance = 1。(10, 0.05) 是角落格，鄰居只有 3 個：
  (10, 0.04)、(20, 0.04)、(20, 0.05)。其中一個就是 2.0 的尖峰，
  所以它的平原分數被拉高到 0.642。這說明了排除自身並不能排除「鄰居是尖峰」的情況，
  尤其角落格的分母很小，更容易受影響。

### 2.11 spikiness（`plateau_summary`）

$$
\text{spikiness} = \frac{m(\text{best}) - \bar m_{\mathcal N(\text{best})}}{s_{\mathcal N(\text{best})}}
$$

- 分子：best 的 metric 減去其鄰居的平均（`best_neighborhood_mean`）。
- 分母：鄰居 metric 的**樣本標準差，`ddof=1`**（`best_neighborhood_std`），不包含 best 本身。
- 鄰居少於 2 個，或鄰居 metric 在相對 $10^{-12}$ 內相同（零離散）時，回傳 `None`；後者的
  `best_neighborhood_std` 回報 0.0，而不是平均值捨入造成的約 $10^{-16}$。
- 模組 docstring 把它定位為**次要指標**，因為分母只由少數鄰居估計，數值不穩定。
- 實測：A = (1.384 − 1.056)/0.164 = 1.997；B = (2.000 − 0.039)/0.224 = 8.752。

### 2.12 forest 上排：各組參數的 Sharpe 區間（`interval_table_outcome`，`comparison.py:394-472`）

`interval_table`（`comparison.py:365-391`）呼叫 `interval_table_outcome`：重疊期數不足時整張表回傳 `None`，其他無法計算的列轉成 `None`。

1. 每組參數的日報酬：`period_returns(equity_curve, timestamps, "daily")`（`scripts/demo_statistics.py` 的 `plot_comparison_forest()`）。
2. `align_returns`（`comparison.py:156-200`）用 `pd.concat(..., join="inner")` 取三組都有的日期（`:192`）。
   無時區與有時區的索引混用時直接 raise `ValueError`，指出是時區不一致，而不是靜默得到重疊 0 期。
   標題的「366 common days」是 `overlap`（`scripts/demo_statistics.py` 的 `plot_comparison_forest()`）。
3. **三組一起重抽樣**：用同一個 `StationaryBootstrap(block_size, *columns, ...)`（`comparison.py:446-448`），
   每次重抽樣對三組序列使用**相同的日期索引**。
4. block size（`_block_size`，`comparison.py:505-514`）：對**每一組**分別算 Politis–White 估計值，取**最大值**後無條件進位，
   至少為 1。完全沒有變動的序列不參與估計；所有估計值都不是有限數時，$b = 1$。實測 $b = 2$。
5. 統計量為 `partial(sharpe_ratio, periods_per_year=365.0)`，`reps = 2000`（`COMPARISON_REPS`）、
   `seed = 0`（`scripts/demo_statistics.py` 的 `plot_comparison_forest()`）。
6. 區間用百分位數法 $[Q_{0.025}, Q_{0.975}]$，每組各自丟棄 NaN 後計算（`comparison.py:454-461`）。

| 標籤 | 點估計 | 95% CI | valid_reps |
| --- | ---: | --- | ---: |
| KD(9, 3, 3) | −6.67 | [−8.67, −4.78] | 2000 |
| KD(14, 3, 3) | −7.57 | [−9.20, −5.82] | 2000 |
| KD(21, 5, 5) | −6.00 | [−7.57, −4.58] | 2000 |

KD(14, 3, 3) 的區間和 `kd_inference.md` 的 $[-9.18, -5.87]$ 不同，但點估計相同（−7.57）。
原因是 reps（2000 對 1000）不同，而且這裡是三組共同重抽樣，亂數消耗方式也不同。
兩邊的 block size 實測都是 2。

### 2.13 forest 下排：配對差異（`difference_interval_outcome`，`comparison.py:239-362`）

`difference_interval`（`comparison.py:203-236`）呼叫 `difference_interval_outcome`，並把所有 `IntervalUnavailable` 轉成 `None`。

**對同一組索引重抽樣**：先以 `align_returns({"a": a, "b": b})` 對齊日期（`:319`），再用
`StationaryBootstrap(block_size, values_a, values_b, seed=np.random.default_rng(seed))`（`:341-343`）。
arch 對所有位置參數套用**同一組索引**，所以每次重抽樣中 $a$、$b$ 取到的是同一批日期，保留了兩者的相關性。
每一對都會重新以 `seed = 0` 建立亂數產生器（`scripts/demo_statistics.py` 的 `plot_comparison_forest()`）。

**block size：從差額估計**（`_difference_block_size`、`_jackknife_influence`，`comparison.py:517-543`；docstring `:258-289`）。
區塊要保留的是「差額估計量」的時間相依性，不是 $a$、$b$ 各自的相依性。兩策略共享較大的市場雜訊時，雜訊在差額中相消，
留下的小而持續的差額才是要保留的結構；若照 $a$、$b$ 各自估計，區塊長度會跟著短記憶的雜訊走，低估差額的變異，
把不顯著的差額判成顯著。因此先算差額的 jackknife 影響值

$$
I_t = (n-1)\big(\hat\Delta - \hat\Delta_{(-t)}\big),\qquad
\hat\Delta_{(-t)} = \widehat{SR}_{\text{ann}}(a_{(-t)}) - \widehat{SR}_{\text{ann}}(b_{(-t)}),
$$

$a_{(-t)}$、$b_{(-t)}$ 是兩序列同時去掉第 $t$ 期。對平均數而言 $I_t$ 就是 $a_t - b_t$ 減去常數。
再對 $I_t$ 序列算 Politis–White 估計值（含 Patton–Politis–White 修正），無條件進位，至少為 1。
任一 leave-one-out 統計量無定義時，改用原始差額 $a - b$ 估計。

這需要額外 $2n$ 次統計量計算。呼叫端可用 `block_size` 參數直接指定區塊長度並略過這一步；
中位數、分位數、最大回撤這類**不平滑**的統計量，jackknife 無法代表其敏感度，**必須**手動指定。
不指定時，估計出的區塊可能太短：區間會變窄、p 值變小，正是這個檢定要防止的假陽性；區塊偏長只損失一些精確度。

$$
\hat\Delta = \widehat{SR}_{\text{ann}}(a) - \widehat{SR}_{\text{ann}}(b),\qquad
\Delta^{*}_j = \widehat{SR}_{\text{ann}}(a^{*}_j) - \widehat{SR}_{\text{ann}}(b^{*}_j),\ \ j=1,\dots,B
$$

區間是 $[Q_{0.025}(\Delta^*), Q_{0.975}(\Delta^*)]$（百分位數法，`:348`），`reps = 2000`。
$B$ 是丟棄 NaN 後的有效重抽樣數。$Q_p$ 是 `np.quantile` 預設的線性內插：把排序後的 $\Delta^*_{(0)},\dots,\Delta^*_{(B-1)}$
在位置 $(B-1)p$ 內插。

**p 值：反轉同一個百分位數區間**（`_percentile_p_value`、`_crossing`，`comparison.py:546-578`；docstring `:291-300`）：

$$
p = \min\!\Big(1,\ 2\cdot\min\big\{\,\sup\{q : Q_q(\Delta^*) \le 0\},\ \ 1 - \inf\{q : Q_q(\Delta^*) \ge 0\}\,\big\}\Big)
$$

也就是讓上述內插區間剛好排除 0 的最小 $1-\text{confidence}$。區間和 p 值用的是同一個內插分位數函數，
所以「區間不含 0」恰好等於「$p < 1-\text{confidence}$」，對任何 reps 與 confidence 都成立
（只有兩者剛好相等時，可能受浮點捨入影響）。所有抽樣都在 0 的同一側時 $p = 0$。

⚠ **有限重抽樣的限制**：$p < 2/(B-1)$ 時，0 落在最極端的兩個抽樣之間，p 值只是這兩點之間的內插，
reps 很少時並不精確。要判讀的顯著水準應遠高於 $2/(B-1)$；本 demo 的 $B = 2000$，約 0.001。

⚠ **舊定義**：先前的 p 值用離散尾端計數
$2\min\{\tfrac1B\sum_j \mathbf 1[\Delta^*_j \le 0],\ \tfrac1B\sum_j \mathbf 1[\Delta^*_j \ge 0]\}$，
區間卻用線性內插，所以兩者在 reps 少時可能矛盾（例如 reps = 20 時 95% 區間全為正，p 卻是 0.1）；這一定義已移除。

⚠ **差異**：教科書的 bootstrap 假設檢定通常先把分配平移到虛無假設下（例如以 $\Delta^*-\hat\Delta$ 為基準），
再看 $\hat\Delta$ 的位置。這裡直接看**未平移**的 $\Delta^*$ 分配中 0 落在哪個分位，屬於「反轉百分位數區間」的做法。

⚠ **下表數字尚未重新產生**：下表與 `comparison_forest.png` 仍是改用差額 block size 與新 p 值定義**之前**的輸出
（當時 block size 取兩序列估計值的較大者）。重跑 `scripts/demo_statistics.py` 後會更新。

**rho**（`_correlation`，`comparison.py:581-587`；欄位說明 `:135-136`）：

$$
\rho = \mathrm{corr}_{\text{Pearson}}\big(r^{a}_d,\ r^{b}_d\big),\qquad d \in \text{兩組共同日期}
$$

它是兩組參數**對齊後的原始日報酬**（未重抽樣）的 Pearson 相關係數，用 `np.corrcoef` 計算。
它不是 bootstrap 差異的相關係數，也不是權益曲線的相關係數。任一序列完全不變時為 `None`。

| 配對 | $\hat\Delta$ | 95% CI | p | rho |
| --- | ---: | --- | ---: | ---: |
| KD(9, 3, 3) − KD(14, 3, 3) | +0.89 | [−0.85, +2.68] | 0.318 | 0.68 |
| KD(9, 3, 3) − KD(21, 5, 5) | −0.67 | [−2.94, +1.70] | 0.542 | 0.34 |
| KD(14, 3, 3) − KD(21, 5, 5) | −1.56 | [−3.57, +0.48] | 0.120 | 0.36 |

**顏色**（`scripts/demo_statistics.py` 的 `plot_comparison_forest()`）：區間不含 0 用黑色（`INK`），含 0 用灰色（`MUTED`）。
三個差異的區間都含 0，所以都是灰色。

---

## 三、真實與合成資料對照

| 圖／檔案 | 部分 | 性質 | 來源 |
| --- | --- | --- | --- |
| `plateau_demo.png` | 左：案例 A 熱圖數值 | **合成** | `build_cases`，seed 0，$\mathcal N(0, 0.05^2)$ 加上確定性的錐形 |
| `plateau_demo.png` | 右：案例 B 熱圖數值 | **合成** | 同上，$\mathcal N(0, 0.3^2)$，加上固定尖峰 2.0 |
| `plateau_demo.png` | 參數軸 lookback／threshold | **虛構** | 常數 `LOOKBACKS`、`THRESHOLDS`，不對應任何策略 |
| `plateau_demo.png` | 白圈、黑框、標題中的 metric_gap／grid_distance | 合成資料，經**正式模組**計算 | `plateau_summary` |
| `plateau_table.md` | 全部 | **合成** | 同上 |
| `kd_equity.png` | 權益曲線、標題起訖權益 | **真實回測** | `cd03bf19459b48af.pkl` |
| `kd_equity.png` | 回撤面積、Max DD | **真實回測**（由權益推導） | 同上，見 2.8 |
| `kd_equity.png` | 標題字串「KD crossover, BTC/USDT 1h」 | 寫死的文字 | `scripts/demo_statistics.py` 的 `plot_kd_equity()` |
| `comparison_forest.png` | 上排三組點估計與 CI | **真實回測**（重新執行）＋ bootstrap | `run_kd_variants`，seed 0 |
| `comparison_forest.png` | 下排差異、CI、p、rho | **真實回測**（重新執行）＋ bootstrap | 同上 |
| `comparison_forest.png` | 「366 common days」 | **真實回測** | `align_returns` 的 overlap |
| `comparison_forest.png` | 標題「BTC/USDT 1h」、註腳「2024」 | 寫死的文字 | `scripts/demo_statistics.py` 的 `plot_comparison_forest()` |
| `kd_inference.md` | 日／月數、Sharpe、CI、PSR、t、勝率、換手率 | **真實回測** | `cd03bf19459b48af.pkl` |
| `kd_inference.md` | 成本拆解 | 交易紀錄推得的 `traded_notional`（`summarize_breakeven`）× **寫死的**費率 | 估計值，見 2.7 |

---

## 四、已知限制

以下內容整理自程式碼的 docstring；最後幾點是本次檢查中實際發現的問題。

1. **樣本內與樣本外**：`comparison.py:21-23` 指出，比較時只能使用樣本外報酬（例如 walk-forward 測試區間）。
   用調校過參數的樣本內報酬比較，量到的是哪個策略「過度擬合得更徹底」，而不是哪個策略比較好。
   三張 KD 相關結果都來自 2024 全年的**樣本內**報酬；forest 圖註腳也有標明（`scripts/demo_statistics.py` 的 `plot_comparison_forest()`）。
2. **三組參數未經調校**：forest 圖註腳寫的是「untuned parameter sets」。
   KD(9, 3, 3)、KD(14, 3, 3)、KD(21, 5, 5) 是程式中手動列出的常見組合（`scripts/demo_statistics.py` 的常數 `KD_VARIANTS`），
   沒有經過最佳化或搜尋，所以不能當作「最佳參數」的證據，也不代表 KD 策略的參數空間。
3. **CI 重疊不代表不顯著**：`comparison.py:17-20` 與 `interval_table_outcome` docstring `:409-413` 指出，
   各自的區間忽略了策略之間的相關性，只能當作保守的初步篩選。區間不重疊可以推得差異顯著，
   但區間重疊**不能**推得不顯著，正式判斷要看配對差異。
4. **`cost_per_turnover` 為何低估**：`breakeven.py:32-34` 指出，成本是交易紀錄 `fee` 欄位的加總。
   引擎若把成本併入 `net_pnl`、卻沒有寫進 `fee`，成本就會被低估。合併前的引擎把 `fee` 寫死為 0，
   所以這份 pickle 的 `total_cost` 與 `cost_per_turnover` 都是 0，完全看不到成本；
   `kd_inference.md` 因此改用 2.7 的外部估計。合併後的引擎會寫入手續費，但 signal 模式的 slippage
   仍內含於成交價（`catalog.py:233-234`），所以 `cost_per_turnover` 仍會少算 slippage 那一部分。
5. **平原分析不能取代樣本外驗證**：`plateau.py` 模組 docstring 的 Limitation 指出，當整個參數區域都對同一種市場型態過度擬合時，
   所有鄰居會因為相同的原因看起來都很好，形成假平原。
6. **平原分析的設計取捨**（`plateau.py` 模組 docstring 的 First-version design decisions）：網格不均勻時，半徑 1 對應的實際參數變化量並不固定；
   參數超過 3 個時，鄰域最多有 $3^d-1$ 個點，已不算局部，會以 `dimension_warning` 標示；
   spikiness 的分母是少數鄰居的樣本標準差，數值不穩定。
7. **統計顯著不等於經濟顯著**：`comparison.py:27-29` 指出，估計很精確但很小的優勢，可能撐不過成本或容量限制。
8. **期間、標的與成本必須一致**：`comparison.py:24-26` 指出，這是實驗設計的問題，統計方法無法修補；
   程式只保證取共同期間。
9. **小時與日頻 Sharpe**：`sharpe.py:7-13` 指出，小時報酬有自相關，用 $\sqrt{P}$ 年化會有偏差。
10. **最後一天只有 1 根 K 棒**：2024-12-31 的「日報酬」只涵蓋 1 小時，但仍算成 1 個觀測值（見 2.1）。
11. **資料來源與 market_type 是推定的**：pickle 沒有記錄這兩項（見第一節）。
12. **`latest_cache()` 不檢查內容**：它只挑 mtime 最新的 pickle。之後只要產生任何較新的 pickle，
    `kd_equity.png` 與 `kd_inference.md` 就會改用那份資料，而圖的標題文字仍然寫著 KD／BTC/USDT 1h。
13. **`total_volume` 在合併前後的定義不同**：見 2.7 第 4 點。腳本因此改從交易紀錄推導成本，
    不依賴該欄位。

---

## 五、可重現性

### 固定的 seed 與常數

| 用途 | 值 | 位置 |
| --- | --- | --- |
| 平原合成資料 | `np.random.default_rng(0)`，先抽 $\mathcal N(0,0.05^2)$ 再抽 $\mathcal N(0,0.3^2)$ | `scripts/demo_plateau.py:35, 54-57`（`demo_statistics.py` 的 `SEED` 與 `plateau_cases()` 相同） |
| `kd_inference.md` 的 Sharpe CI | `seed=0`、`reps=1000`、confidence 0.95 | `scripts/demo_statistics.py` 的 `SEED`、`BOOTSTRAP_REPS`、`write_kd_inference()` |
| forest 上排 `interval_table` | `seed=0`、`reps=2000`、confidence 0.95 | `scripts/demo_statistics.py` 的 `COMPARISON_REPS`、`SEED`、`plot_comparison_forest()` |
| forest 下排 `difference_interval` | 每一對都使用 `seed=0`、`reps=2000` | `scripts/demo_statistics.py` 的 `plot_comparison_forest()` |
| 年化期數 | 365.0 | `scripts/demo_statistics.py` 的 `DAILY_PERIODS_PER_YEAR` |
| 成本估計費率 | fees 0.0006、slippage 0.0005 | `scripts/demo_statistics.py` 的 `FEES`、`SLIPPAGE` |
| 平原半徑 | 1 | `plateau.py` 的 `DEFAULT_RADIUS` |
| bootstrap 最少觀測數 | 20 | `sharpe.py:45` |
| 函式庫版本（目前） | `arch` 8.0.0、`vectorbt` 1.2.0、`pandas` 3.0.6、`numpy` 2.5.4、`quantstats` 0.0.86 | 目前的 `uv.lock` |
| 函式庫版本（產生 demo 圖時） | `arch` 8.0.0、`vectorbt` 1.0.0、`pandas` 2.3.3、`numpy` 2.2.6 | 合併前（`814b05a`）的 `uv.lock` |

重現指令以目前的 `uv.lock` 建立環境，所以重現時使用的是「目前」那一列的版本，不是產生 demo 圖時的版本。
第 2.13 節的表格與 `comparison_forest.png` 尚未以目前程式重新產生（見該節）。

### 重現指令

所有指令都在 repository 根目錄執行，並先完成 `uv sync --locked --all-extras`。

**`kd_equity.png`、`comparison_forest.png`、`plateau_gridindex.png`（以及 `kd_inference.md`、
`plateau_gridindex_table.md`）**

```bash
uv run python scripts/demo_statistics.py
```

**`plateau_demo.png`（以及 `plateau_table.md`）**

```bash
uv run python scripts/demo_plateau.py
```

注意事項：

1. 兩支腳本的輸出檔名不重疊，執行順序不影響結果。`demo_statistics.py` 的網格索引版（軸標籤為網格索引、
   配色為 `Blues`）寫入 `plateau_gridindex.png` 與 `plateau_gridindex_table.md`；lookback／threshold 版的
   `plateau_demo.png` 與 `plateau_table.md` 只由 `demo_plateau.py` 產生。
2. `kd_equity.png` 與 `kd_inference.md` 讀取的是 mtime 最新的 pickle。重現前請確認
   `workspace/data/processed/cache/` 中最新的仍是 `cd03bf19459b48af.pkl`（或與它相同內容的
   `e9d1e13c2eb25909.pkl`）。依目前程式碼，`BacktestService.run_raw_config` 不會寫入這個目錄：
   本次重跑三組回測前後，目錄內容都沒有改變。
3. `comparison_forest.png` 會用目前的程式碼重新執行三次回測，需要能取得 BTC/USDT 1h 的 2024 資料
   （本機已有 `workspace/data/processed/btcusdt_1h_2024.parquet`）。本次用合併後的程式碼重跑，
   圖上所有標註數字都與現有圖檔一致（見 2.12、2.13）。但本次沒有實際重新產生 PNG 檔，
   也沒有比對影像位元組。
4. 平原圖完全使用合成資料，不需要網路或快取；在固定 seed 下結果是確定的。
