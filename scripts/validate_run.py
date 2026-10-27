"""Validate a TradingDev run's metrics against its raw series.

Usage, from the repository root:
    uv run python scripts/validate_run.py [path/to/pipeline_result.pkl]

Without a path, it checks the last ``workspace/runs/*/pipeline_result.pkl``
in name order, or, when there is none, the last
``workspace/data/processed/cache/*.pkl`` in name order; it does not look at
modification times, and the paths are relative to the current directory,
ignoring ``TRADINGDEV_WORKSPACE`` and ``TRADINGDEV_DATA_ROOT``. Only simple backtests
are supported. The file is unpickled, which can run arbitrary code, so only
check pickles you trust.

The checked metrics are treated as claims and recomputed from the equity
curve, timestamps and trades, following the engine's definitions in
``src/tradingdev/domain/performance/catalog.py``: total_pnl, total_return,
max_drawdown, total_trades, win_rate, profit_factor, n_days, n_months,
daily_pnl_mean, daily_pnl_std, monthly_pnl_mean and sharpe_ratio. Other
metrics are only checked for being finite or ``None`` with a reason. Each
check ends as:

* ``PASS``: the reported value matches the recomputation.
* ``FAIL``: it does not, or the record is missing a field or is corrupt.
* ``N/A``: the engine legitimately reports no value, such as a win rate
  without closed trades; the reason code from the run's metric metadata
  (``no_trades``, ``zero_denominator``, ``unbounded``, ``not_applicable``,
  ``missing_annualization``, ...) is shown. The reason is checked against
  what the run itself implies (the trades, the mode, the capital basis, the
  bar frequency and the annualisation setting), so a wrong reason fails.
* ``WARN``: advisory, not an error, such as gaps between bars, which the
  engine allows.

Exit code: 0 without FAIL, 1 with any FAIL (an explicit path that cannot be
read is a FAIL), 2 when the file is not a simple backtest or, without a path,
no file is found.
"""

from __future__ import annotations

import glob
import math
import numbers
import pickle
import sys
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from tradingdev.domain.performance.catalog import METRIC_CATALOG
from tradingdev.domain.performance.sampling import (
    DAILY_EQUITY_METRICS,
    daily_observation_unavailable_reason,
)
from tradingdev.domain.statistics.breakeven import summarize_breakeven

if TYPE_CHECKING:
    from collections.abc import Callable

TOL = 1e-6  # relative tolerance for recomputed values
PERIOD_PNL_STD_DDOF = 0  # catalog: daily/monthly P&L std uses ddof=0

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

# Check names, shared by each check and by the guard that fails it.
A1 = "A1 total_pnl = 期末權益 − 初始權益"
A2 = "A2 交易淨損益總和（含未平倉）= total_pnl"
A3 = "A3 total_trades = 已平倉交易筆數"
A4 = "A4 win_rate 可重現（已平倉）"
A5 = "A5 profit_factor 可重現（已平倉）"
A6 = "A6 max_drawdown 可重現"
A7 = "A7 total_return = total_pnl / 初始資金"
T1 = "T1 n_days = 實際 UTC 日曆天數"
T2 = "T2 n_months = 實際 UTC 月數"
T3 = "T3 daily_pnl_std 為真正的日波動"
T4 = "T4 monthly_pnl_mean 為真正的月均"
T5 = "T5 daily_pnl_mean 為真正的日均"
T6 = "T6 sharpe_ratio 與日頻重算相符"
V2 = "V2 win_rate 落在 [0,1]"
V3 = "V3 max_drawdown 為非負幅度（catalog 定義）"
C2 = "C2 交易紀錄可推算成交名目"

results: list[tuple[str, str, str]] = []


class CorruptRecordError(Exception):
    """The run lacks a field or holds a value its contract does not allow."""


class Run:
    """What the checks read from one simple backtest."""

    # A plain class: a dataclass needs the module registered in sys.modules,
    # which loading this script by path does not do.
    def __init__(
        self,
        *,
        metrics: dict[str, Any],
        unavailable: dict[str, str],
        settings: dict[str, Any],
        config: dict[str, Any],
        mode: str,
        equity: pd.Series,
        trades: list[dict[str, Any]],
    ) -> None:
        self.metrics = metrics
        self.unavailable = unavailable
        self.settings = settings
        self.config = config
        self.mode = mode
        self.equity = equity
        self.trades = trades

    def metric(self, key: str) -> float | None:
        """The reported value, or ``None`` when the engine gives a reason."""
        if key not in self.metrics:
            raise CorruptRecordError(f"metrics 缺少 {key}")
        value = self.metrics[key]
        if value is None:
            if not self.unavailable.get(key):
                raise CorruptRecordError(f"{key} 為 None，但 metadata 沒有原因")
            return None
        if isinstance(value, bool) or not isinstance(value, numbers.Real):
            raise CorruptRecordError(f"{key} 不是數值：{value!r}")
        return float(value)

    def setting(self, key: str, config_key: str | None = None) -> Any:
        """An effective engine setting; older runs only kept the config."""
        if key in self.settings:
            return self.settings[key]
        return self.config.get(config_key or key)

    @property
    def initial(self) -> float:
        """Equity before the first bar: the capital, or zero P&L in volume mode."""
        if self.mode == "volume":
            return 0.0
        capital = self.setting("initial_cash", "init_cash")
        if capital is None:
            raise CorruptRecordError("signal 模式缺少初始資金")
        return float(capital)

    def expected_reason(self, key: str) -> str | None:
        """The reason the engine must give instead of a value, from the saved data.

        Mirrors the order of the engine's checks in
        ``domain/backtest/metrics.py``: mode, capital basis, daily sampling,
        then annualisation. ``None`` means the metric must have a value, as
        far as these rules go; statistics that turn out undefined on the
        data are left to each check.
        """
        definition = METRIC_CATALOG[key]
        if self.mode not in definition.modes:
            return "not_applicable"
        if definition.provider == "quantstats":
            equity = self.equity.to_numpy()
            baseline = np.concatenate(([self.initial], equity))
            if not (np.all(baseline[:-1] > 0) and np.all(equity >= 0)):
                return "nonpositive_capital"
        if key in DAILY_EQUITY_METRICS:
            frequency = self.setting("frequency", "timeframe")
            if frequency is not None:
                reason = daily_observation_unavailable_reason(str(frequency))
                if reason is not None:
                    return reason
        if (
            definition.requires_annualization
            and self.setting("periods_per_year") is None
        ):
            return "missing_annualization"
        return None

    @property
    def closed(self) -> list[dict[str, Any]]:
        # The catalog counts closed trades only; records without a status
        # predate it and are closed trades.
        return [t for t in self.trades if t.get("status", "closed") == "closed"]


def check(name: str, ok: bool | None, detail: str = "") -> None:
    status = "WARN" if ok is None else ("PASS" if ok else "FAIL")
    results.append((status, name, detail))


def not_available(name: str, reason: str) -> None:
    results.append(("N/A", name, f"引擎無值，原因 {reason}"))


def guarded(name: str, body: Callable[[], None]) -> None:
    """Run one check; a missing field or bad record fails it, never crashes."""
    try:
        body()
    except CorruptRecordError as exc:
        check(name, False, f"資料缺漏或損壞：{exc}")
    except (AttributeError, KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
        check(name, False, f"無法重算：{type(exc).__name__}: {exc}")


def close(a: float, b: float, tol: float = TOL) -> bool:
    scale = max(abs(a), abs(b), 1.0)
    return abs(a - b) / scale <= tol


def compare(
    run: Run,
    name: str,
    key: str,
    expected: float | str,
    tol: float = TOL,
    note: str = "",
) -> None:
    """Compare a metric with a recomputation, or with the reason it has none.

    ``expected`` is the recomputed value, or the reason code the raw data
    implies the engine must give instead of a value.
    """
    value = run.metric(key)
    if value is None:
        reason = run.unavailable[key]
        if isinstance(expected, str) and expected != reason:
            check(name, False, f"引擎原因 {reason}，但原始資料推得 {expected}")
        elif isinstance(expected, str):
            not_available(name, reason)
        else:
            check(name, False, f"引擎原因 {reason}，但原始資料可算出 {expected:.4f}")
    elif isinstance(expected, str):
        check(name, False, f"報告 {value:.4f}，但原始資料推得無值（{expected}）")
    else:
        check(
            name,
            close(value, expected, tol),
            f"報告 {value:.4f} vs 重算 {expected:.4f}{note}",
        )


def period_pnl(run: Run, rule: str) -> pd.Series:
    """Closing equity of each observed UTC period minus the previous close.

    The first period is measured from the initial equity, matching the
    engine's P&L, which starts from the capital rather than the first bar.
    """
    closes = run.equity.resample(rule).last().dropna()
    return closes - closes.shift(1, fill_value=run.initial)


def daily_returns(run: Run) -> pd.Series:
    closes = run.equity.resample("D").last().dropna()
    return closes / closes.shift(1, fill_value=run.initial) - 1.0


def load(path: str) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def main(path: str) -> int:
    results.clear()
    try:
        pr = load(path)
    except Exception as exc:  # noqa: BLE001 - any unreadable file is one FAIL
        check("L1 讀取 pickle", False, f"{type(exc).__name__}: {exc}")
        return report()
    if (
        getattr(pr, "mode", None) != "simple"
        or getattr(pr, "backtest_result", None) is None
    ):
        print(f"mode={getattr(pr, 'mode', None)}: 只支援 simple backtest")
        return 2
    print(f"file      : {path}")
    try:
        return validate(pr)
    except Exception as exc:  # noqa: BLE001 - a corrupt run is a FAIL, not a crash
        check("X1 檢查中止", False, f"資料無法讀取：{type(exc).__name__}: {exc}")
        return report()


def validate(pr: Any) -> int:
    bt = pr.backtest_result
    config = dict((pr.config_snapshot or {}).get("backtest") or {})
    metadata = getattr(bt, "metric_metadata", None) or {}
    eq = np.asarray(bt.equity_curve, dtype=float)
    trades = list(bt.trades)
    mode = str(getattr(bt, "mode", config.get("mode", "signal")))
    print(f"mode      : {mode}   bars: {len(eq)}   trades: {len(trades)}")

    # ---------- 1. structure ----------
    ts_raw = getattr(bt, "timestamps", None)
    check("S1 timestamps 存在", ts_raw is not None)
    if ts_raw is None:
        return report()
    ts = pd.DatetimeIndex(pd.to_datetime(ts_raw, utc=True))
    check("S2 equity 與 timestamps 等長", len(ts) == len(eq), f"{len(ts)} vs {len(eq)}")
    check("S3 時間嚴格遞增且無重複", bool(ts.is_monotonic_increasing and ts.is_unique))
    if len(ts) != len(eq) or len(ts) == 0:
        return report()

    bars_per_day: float | None = None
    diffs = ts.to_series().diff().dropna()
    if diffs.empty:
        check("S4 K 棒間隔一致", None, "只有一根 K 棒，無法判斷間隔")
    elif diffs.median() <= pd.Timedelta(0):
        check(
            "S4 K 棒間隔一致", False, f"間隔中位數 {diffs.median()}，時間戳重複或倒序"
        )
    else:
        step = diffs.median()
        regular = float((diffs == step).mean())
        # The engine allows calendar gaps, so irregular bars only warn.
        check(
            "S4 K 棒間隔一致",
            True if regular > 0.99 else None,
            f"間隔 {step}, 一致比例 {regular:.3%}",
        )
        span_bars = int((ts[-1] - ts[0]) / step) + 1
        check(
            "S5 期間內無缺漏 K 棒",
            True if len(ts) >= span_bars * 0.99 else None,
            f"實際 {len(ts)} / 期望 {span_bars}",
        )
        bars_per_day = pd.Timedelta("1D") / step

    run = Run(
        metrics=dict(bt.metrics),
        unavailable=dict(metadata.get("unavailable") or {}),
        settings=dict(metadata.get("settings") or {}),
        config=config,
        mode=mode,
        equity=pd.Series(eq, index=ts),
        trades=trades,
    )

    # ---------- 2. accounting identities ----------
    def a1() -> None:
        compare(run, A1, "total_pnl", eq[-1] - run.initial)

    def a2() -> None:
        # Marked-to-market: open trades' P&L is already in the equity curve.
        net = float(sum(t["net_pnl"] for t in trades))
        total = run.metric("total_pnl")
        if total is None:
            raise CorruptRecordError("total_pnl 無值")
        check(
            A2,
            close(net, total, 1e-4),
            f"{net:.2f} vs {total:.2f}",
        )

    def a3() -> None:
        reported = run.metric("total_trades")
        open_count = len(trades) - len(run.closed)
        check(
            A3,
            reported == len(run.closed),
            f"{reported:.0f} vs {len(run.closed)}（未平倉 {open_count} 筆不計入）",
        )

    def a4() -> None:
        closed = run.closed
        wins = sum(1 for t in closed if t["net_pnl"] > 0)
        expected = wins / len(closed) if closed else "no_trades"
        compare(run, A4, "win_rate", expected)

    def a5() -> None:
        closed = run.closed
        gain = sum(t["net_pnl"] for t in closed if t["net_pnl"] > 0)
        loss = -sum(t["net_pnl"] for t in closed if t["net_pnl"] < 0)
        expected: float | str = (
            "no_trades"
            if not closed
            else gain / loss
            if loss
            else "unbounded"
            if gain
            else "zero_denominator"
        )
        compare(run, A5, "profit_factor", expected)

    def a6() -> None:
        reason = run.expected_reason("max_drawdown")
        if reason is not None:
            compare(run, A6, "max_drawdown", reason)
            return
        equity = np.concatenate(([run.initial], eq))
        drawdown = float(np.max(1.0 - equity / np.maximum.accumulate(equity)))
        value = run.metric("max_drawdown")
        if value is None:
            compare(run, A6, "max_drawdown", drawdown)
            return
        # Only the magnitude here; V3 checks the catalog's nonnegative sign.
        check(
            A6,
            close(abs(value), drawdown),
            f"報告 {value:.4f} vs 重算 {drawdown:.4f}（比較幅度）",
        )

    def a7() -> None:
        reason = run.expected_reason("total_return")
        if reason is not None:
            compare(run, A7, "total_return", reason)
            return
        expected = (eq[-1] - run.initial) / run.initial
        compare(run, A7, "total_return", expected)

    for name, body in (
        (A1, a1),
        (A2, a2),
        (A3, a3),
        (A4, a4),
        (A5, a5),
        (A6, a6),
        (A7, a7),
    ):
        guarded(name, body)

    # ---------- 3. time-scale (catches the mislabelled daily stats) ----------
    def t1() -> None:
        true_days = int(ts.normalize().nunique())
        hint = ""
        reported = run.metrics.get("n_days")
        if (
            isinstance(reported, (int, float))
            and reported != true_days
            and abs(reported - len(eq)) <= 1
            and bars_per_day is not None
        ):
            hint = f"  ← 疑似回報 K 棒根數（每日 {bars_per_day:.0f} 根）"
        compare(run, T1, "n_days", true_days, note=hint)

    def t2() -> None:
        true_months = int(ts.tz_localize(None).to_period("M").nunique())
        compare(run, T2, "n_months", true_months)

    def t3() -> None:
        reason = run.expected_reason("daily_pnl_std")
        if reason is not None:
            compare(run, T3, "daily_pnl_std", reason)
            return
        # The catalog fixes ddof=0; the run's own metadata is not trusted for it.
        declared = run.settings.get("period_pnl_std_ddof", PERIOD_PNL_STD_DDOF)
        if declared != PERIOD_PNL_STD_DDOF:
            raise CorruptRecordError(
                f"metadata 宣稱 period_pnl_std_ddof={declared!r}，"
                f"catalog 定義為 {PERIOD_PNL_STD_DDOF}"
            )
        ddof = PERIOD_PNL_STD_DDOF
        daily_std = float(period_pnl(run, "D").std(ddof=ddof))
        reported = run.metrics.get("daily_pnl_std")
        hint = ""
        if (
            isinstance(reported, (int, float))
            and reported > 0
            and not close(reported, daily_std)
            and bars_per_day is not None
        ):
            hint = (
                f"（重算/報告 = {daily_std / reported:.2f}，"
                f"√(每日根數) = {np.sqrt(bars_per_day):.2f}，"
                "相符即代表報告值其實是每根 K 棒）"
            )
        compare(
            run,
            T3,
            "daily_pnl_std",
            daily_std,
            note=f"，ddof={ddof}{hint}",
        )

    def t4() -> None:
        reason = run.expected_reason("monthly_pnl_mean")
        monthly_mean = float(period_pnl(run, "MS").mean())
        compare(run, T4, "monthly_pnl_mean", reason or monthly_mean)

    def t5() -> None:
        reason = run.expected_reason("daily_pnl_mean")
        daily_mean = float(period_pnl(run, "D").mean())
        compare(run, T5, "daily_pnl_mean", reason or daily_mean)

    def t6() -> None:
        saved = run.setting("periods_per_year")
        periods = float(saved) if saved is not None else 365.0
        rate = float(run.setting("risk_free_rate") or 0.0)
        if mode not in METRIC_CATALOG["sharpe_ratio"].modes:
            compare(run, T6, "sharpe_ratio", "not_applicable")
            return

        def excess() -> pd.Series:
            return daily_returns(run) - math.expm1(math.log1p(rate) / periods)

        def annualised(values: pd.Series) -> float:
            return float(values.mean() / values.std(ddof=1) * math.sqrt(periods))

        # Diagnostic only: the value an engine annualising bar returns gives.
        def bar_hint() -> str:
            bars = run.equity.pct_change().dropna()
            if bars_per_day is None or len(bars) < 2 or not bars.std() > 0:
                return ""
            bar_sharpe = bars.mean() / bars.std() * math.sqrt(bars_per_day * periods)
            return f"若以 bar 頻年化會是 {bar_sharpe:.4f}"

        if not run.settings and saved is None:
            # Runs from before metric metadata kept no annualisation setting.
            check(
                T6,
                False,
                "沒有保存 periods_per_year，無法驗證；以 365 試算："
                f"日頻 {annualised(excess()):.4f}，{bar_hint()}，"
                f"報告 {run.metrics.get('sharpe_ratio')}",
            )
            return
        reason = run.expected_reason("sharpe_ratio")
        if reason is not None:
            compare(run, T6, "sharpe_ratio", reason)
            return
        values = excess()
        if len(values) < 2:
            compare(run, T6, "sharpe_ratio", "insufficient_data")
        elif values.std(ddof=1) == 0.0:
            compare(
                run,
                T6,
                "sharpe_ratio",
                "zero_denominator" if values.mean() == 0.0 else "unbounded",
            )
        else:
            compare(
                run,
                T6,
                "sharpe_ratio",
                annualised(values),
                note=(
                    f"（periods_per_year={periods:g}, risk_free_rate={rate:g}；"
                    f"{bar_hint()}）"
                ),
            )

    for name, body in (
        (T1, t1),
        (T2, t2),
        (T3, t3),
        (T4, t4),
        (T5, t5),
        (T6, t6),
    ):
        guarded(name, body)

    # ---------- 4. values & schema ----------
    bad = [
        k
        for k, v in run.metrics.items()
        if isinstance(v, (int, float)) and not np.isfinite(float(v))
    ]
    check(
        "V1 所有指標為有限值或附原因的 None", not bad, f"非有限: {bad}" if bad else ""
    )

    def v2() -> None:
        value = run.metric("win_rate")
        if value is None:
            not_available(V2, run.unavailable["win_rate"])
        else:
            check(V2, 0.0 <= value <= 1.0, f"{value:.4f}")

    def v3() -> None:
        # The catalog reports a nonnegative magnitude with no upper bound:
        # short positions can push equity below zero and the loss past 100%.
        name = V3
        if mode not in METRIC_CATALOG["max_drawdown"].modes:
            compare(run, name, "max_drawdown", "not_applicable")
            return
        value = run.metric("max_drawdown")
        if value is None:
            not_available(name, run.unavailable["max_drawdown"])
        else:
            check(name, value >= 0.0, f"{value:.4f}")

    guarded(V2, v2)
    guarded(V3, v3)
    check("V4 equity 無 NaN/inf", bool(np.isfinite(eq).all()))
    missing = [k for k in REPORT_KEYS if k not in run.metrics]
    check("V5 報表所需欄位齊備", not missing, f"缺少: {missing}" if missing else "")

    # ---------- 5. cost sanity ----------
    def cost() -> None:
        fee_rate = float(config.get("fees", 0.0)) + float(config.get("slippage", 0.0))
        if fee_rate > 0 and trades:
            # total_volume already counts both legs, so derive the cost from
            # the trades' traded notional.
            notional = summarize_breakeven(
                trades, start=ts[0], end=ts[-1]
            ).traded_notional
            total = eq[-1] - run.initial
            estimate = fee_rate * notional
            share = estimate / max(abs(total), 1e-9)
            print(
                f"\n成本拆解  : 估計成本 {estimate:,.0f} / "
                f"毛損益 {total + estimate:,.0f} / "
                f"淨損益 {total:,.0f}  → 成本占比 {share:.0%}"
            )
        if trades and all(float(t.get("fee", 0.0)) == 0.0 for t in trades):
            check(
                "C1 交易明細的 fee 欄位有值",
                None,
                "全為 0，但設定有成本 → fee 欄位未填（淨損益仍已扣除）"
                if fee_rate > 0
                else "設定本身無成本",
            )

    guarded(C2, cost)
    return report()


def report() -> int:
    width = max((len(n) for _, n, _ in results), default=0) + 2
    print()
    for status, name, detail in results:
        print(f"[{status:4}] {name:<{width}} {detail}")
    counts = {
        s: sum(1 for r, _, _ in results if r == s) for s in ("FAIL", "WARN", "N/A")
    }
    print(
        f"\n合計: {len(results)} 項檢查, {counts['FAIL']} FAIL, "
        f"{counts['WARN']} WARN, {counts['N/A']} N/A"
    )
    return 1 if counts["FAIL"] else 0


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
