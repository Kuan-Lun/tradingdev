"""Validate a TradingDev run's metrics against its raw series.

Usage:
    uv run python validate_run.py [path/to/pipeline_result.pkl]

Every reported metric is treated as a claim and recomputed from
equity_curve / timestamps / trades.  Exit code 1 if any check FAILs.
"""

from __future__ import annotations

import glob
import pickle
import sys

import numpy as np
import pandas as pd

TOL = 1e-6  # relative tolerance for accounting identities
LOOSE = 0.02  # 2% tolerance for statistics with convention differences

REPORT_KEYS = [
    "total_pnl",
    "total_return",
    "annual_return",
    "max_drawdown",
    "sharpe_ratio",
    "win_rate",
    "profit_factor",
    "total_trades",
    "total_volume",
    "daily_pnl_mean",
    "daily_pnl_std",
    "daily_pnl_min",
    "daily_pnl_max",
    "daily_pnl_median",
    "n_days",
    "monthly_pnl_mean",
    "monthly_pnl_std",
    "monthly_trades_mean",
    "n_months",
]

results: list[tuple[str, str, str]] = []


def check(name: str, ok: bool | None, detail: str = "") -> None:
    status = "WARN" if ok is None else ("PASS" if ok else "FAIL")
    results.append((status, name, detail))


def close(a: float, b: float, tol: float = TOL) -> bool:
    scale = max(abs(a), abs(b), 1.0)
    return abs(a - b) / scale <= tol


def main(path: str) -> int:
    with open(path, "rb") as f:
        pr = pickle.load(f)
    if pr.mode != "simple":
        print(f"mode={pr.mode}: this checker handles simple backtests only")
        return 2

    bt = pr.backtest_result
    m = dict(bt.metrics)
    eq = np.asarray(bt.equity_curve, dtype=float)
    trades = list(bt.trades)
    cfg = (pr.config_snapshot or {}).get("backtest", {})
    mode = getattr(bt, "mode", cfg.get("mode", "signal"))
    init_cash = getattr(bt, "init_cash", None) or cfg.get("init_cash")

    print(f"file      : {path}")
    print(f"mode      : {mode}   bars: {len(eq)}   trades: {len(trades)}")

    # ---------- 1. structure ----------
    ts_raw = getattr(bt, "timestamps", None)
    check("S1 timestamps 存在", ts_raw is not None)
    if ts_raw is None:
        report()
        return 1

    ts = pd.DatetimeIndex(pd.to_datetime(ts_raw))
    check("S2 equity 與 timestamps 等長", len(ts) == len(eq), f"{len(ts)} vs {len(eq)}")
    check("S3 時間嚴格遞增且無重複", bool(ts.is_monotonic_increasing and ts.is_unique))

    diffs = ts.to_series().diff().dropna()
    step = diffs.median()
    regular = float((diffs == step).mean())
    check("S4 K 棒間隔一致", regular > 0.99, f"間隔 {step}, 一致比例 {regular:.3%}")

    span_bars = int((ts[-1] - ts[0]) / step) + 1
    check(
        "S5 期間內無缺漏 K 棒",
        len(ts) >= span_bars * 0.99,
        f"實際 {len(ts)} / 期望 {span_bars}",
    )

    eqs = pd.Series(eq, index=ts)
    pnl = eqs.diff().dropna()
    bars_per_day = pd.Timedelta("1D") / step

    # ---------- 2. accounting identities ----------
    tot = float(eq[-1] - eq[0])
    check(
        "A1 total_pnl = 期末權益 − 期初權益",
        close(m["total_pnl"], tot),
        f"{m['total_pnl']:.2f} vs {tot:.2f}",
    )

    net = float(sum(t["net_pnl"] for t in trades))
    check(
        "A2 交易淨損益總和 = total_pnl",
        close(net, m["total_pnl"], 1e-4),
        f"{net:.2f} vs {m['total_pnl']:.2f}（差額=未平倉或記帳遺漏）",
    )

    check(
        "A3 total_trades = 交易筆數",
        int(m["total_trades"]) == len(trades),
        f"{m['total_trades']} vs {len(trades)}",
    )

    wins = sum(1 for t in trades if t["net_pnl"] > 0)
    check(
        "A4 win_rate 可重現",
        close(m["win_rate"], wins / max(len(trades), 1), LOOSE),
        f"{m['win_rate']:.4f} vs {wins / max(len(trades), 1):.4f}",
    )

    gp = sum(t["net_pnl"] for t in trades if t["net_pnl"] > 0)
    gl = -sum(t["net_pnl"] for t in trades if t["net_pnl"] < 0)
    pf = gp / gl if gl else float("inf")
    check(
        "A5 profit_factor 可重現",
        close(m["profit_factor"], pf, LOOSE),
        f"{m['profit_factor']:.4f} vs {pf:.4f}",
    )

    dd = eqs / eqs.cummax() - 1.0
    mdd_frac, mdd_abs = float(dd.min()), float((eqs - eqs.cummax()).min())
    ref = mdd_abs if mode == "volume" else mdd_frac
    check(
        "A6 max_drawdown 可重現",
        close(abs(m["max_drawdown"]), abs(ref), LOOSE),
        f"報告 {m['max_drawdown']:.4f} vs 重算 {ref:.4f}（{mode} 模式）",
    )

    if mode != "volume" and init_cash:
        check(
            "A7 total_return = total_pnl / init_cash",
            close(abs(m["total_return"]), abs(tot / init_cash), LOOSE),
            f"{m['total_return']:.4f} vs {tot / init_cash:.4f}",
        )

    # ---------- 3. time-scale (catches the mislabelled daily stats) ----------
    true_days = int(ts.normalize().nunique())
    check(
        "T1 n_days = 實際日曆天數",
        int(m["n_days"]) == true_days,
        f"報告 {m['n_days']} vs 實際 {true_days}"
        + (
            f"  ← 疑似回報 K 棒根數（每日 {bars_per_day:.0f} 根）"
            if abs(int(m["n_days"]) - len(eq)) <= 1
            else ""
        ),
    )

    true_months = int(ts.tz_localize(None).to_period("M").nunique())
    check(
        "T2 n_months = 實際月數",
        int(m["n_months"]) == true_months,
        f"報告 {m['n_months']} vs 實際 {true_months}",
    )

    daily = pnl.resample("D").sum()
    monthly = pnl.resample("ME").sum()
    check(
        "T3 daily_pnl_std 為真正的日波動",
        close(m["daily_pnl_std"], float(daily.std()), 0.05),
        f"報告 {m['daily_pnl_std']:.4f} vs 日頻重算 {daily.std():.4f}"
        f"（日頻/報告 = {daily.std() / max(m['daily_pnl_std'], 1e-12):.2f}，"
        f"√(每日根數) = {np.sqrt(bars_per_day):.2f} "
        f"→ 相符即代表報告值其實是每根 K 棒）",
    )

    check(
        "T4 monthly_pnl_mean 為真正的月均",
        close(m["monthly_pnl_mean"], float(monthly.mean()), 0.05),
        f"報告 {m['monthly_pnl_mean']:.2f} vs 月頻重算 {monthly.mean():.2f}",
    )

    check(
        "T5 日/月加總 = total_pnl",
        close(float(daily.sum()), tot, 1e-4) and close(float(monthly.sum()), tot, 1e-4),
        f"日 {daily.sum():.2f} / 月 {monthly.sum():.2f} vs {tot:.2f}",
    )

    # ---------- 4. Sharpe: which frequency? ----------
    r_bar = eqs.pct_change().dropna()
    sr_bar = float(r_bar.mean() / r_bar.std() * np.sqrt(bars_per_day * 365))
    rd = daily / eqs.resample("D").first()
    sr_day = float(rd.mean() / rd.std() * np.sqrt(365))
    matched = (
        "bar/hourly"
        if close(m["sharpe_ratio"], sr_bar, 0.05)
        else "daily"
        if close(m["sharpe_ratio"], sr_day, 0.05)
        else "無法對應"
    )
    check(
        "T6 sharpe_ratio 的年化頻率可辨識",
        matched != "無法對應",
        f"報告 {m['sharpe_ratio']:.4f} | bar 頻 {sr_bar:.4f} | 日頻 {sr_day:.4f}"
        f" → 使用 {matched}",
    )
    check(
        "T7 sharpe 以日頻為基準",
        None if matched != "daily" else True,
        "回報值來自 bar 頻年化；跨策略比較前應統一為日頻" if matched != "daily" else "",
    )

    # ---------- 5. values & schema ----------
    bad = [
        k
        for k, v in m.items()
        if isinstance(v, (int, float)) and not np.isfinite(float(v))
    ]
    check("V1 所有指標為有限值", not bad, f"非有限: {bad}" if bad else "")
    check("V2 win_rate 落在 [0,1]", 0.0 <= float(m["win_rate"]) <= 1.0)
    check("V3 max_drawdown 非正", float(m["max_drawdown"]) <= 0.0)
    check("V4 equity 無 NaN/inf", bool(np.isfinite(eq).all()))

    missing = [k for k in REPORT_KEYS if k not in m]
    check("V5 報表所需欄位齊備", not missing, f"缺少: {missing}" if missing else "")

    # ---------- 6. cost sanity ----------
    fee_rate = float(cfg.get("fees", 0.0)) + float(cfg.get("slippage", 0.0))
    vol = float(m.get("total_volume", 0.0) or 0.0)
    if fee_rate > 0 and vol > 0:
        cost = 2 * fee_rate * vol
        gross = tot + cost
        print(
            f"\n成本拆解  : 估計成本 {cost:,.0f} / 毛損益 {gross:,.0f} / "
            f"淨損益 {tot:,.0f}  → 成本占比 {cost / max(abs(tot), 1e-9):.0%}"
        )
    if trades and all(float(t.get("fee", 0.0)) == 0.0 for t in trades):
        check(
            "C1 交易明細的 fee 欄位有值",
            None,
            "全為 0，但設定有成本 → fee 欄位未填（淨損益仍已扣除）"
            if fee_rate > 0
            else "設定本身無成本",
        )

    return report()


def report() -> int:
    width = max(len(n) for _, n, _ in results) + 2
    print()
    for status, name, detail in results:
        print(f"[{status:4}] {name:<{width}} {detail}")
    fails = sum(1 for s, _, _ in results if s == "FAIL")
    warns = sum(1 for s, _, _ in results if s == "WARN")
    print(f"\n合計: {len(results)} 項檢查, {fails} FAIL, {warns} WARN")
    return 1 if fails else 0


if __name__ == "__main__":
    if len(sys.argv) > 1:
        target = sys.argv[1]
    else:
        found = sorted(glob.glob("workspace/data/processed/cache/*.pkl")) + sorted(
            glob.glob("workspace/runs/*/pipeline_result.pkl")
        )
        if not found:
            print("找不到 pipeline_result.pkl，請指定路徑")
            raise SystemExit(2)
        target = found[-1]
    raise SystemExit(main(target))
