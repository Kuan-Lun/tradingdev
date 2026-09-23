"""Provider-neutral requirements and independent signal expectations."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Scenario:
    name: str
    parameters: dict[str, int]
    overrides: dict[str, int]
    requirement: str
    repair: bool = False
    legacy: bool = False

    @property
    def strategy_id(self) -> str:
        return f"llm_{self.name}"

    @property
    def prompt(self) -> str:
        if self.legacy:
            preparation = (
                "後端已有這個策略的舊格式檔案。先 list_strategies 找到它，"
                "再 get_strategy 讀取現有原始碼與 YAML，依工具提供的恢復指引處理。"
                "必須先完成這兩次讀取才可 save_strategy，不可跳過舊內容直接建立。"
                "即使舊 metadata 標示 runnable，也必須重新儲存、驗證及 dry-run。"
            )
        elif self.repair:
            preparation = (
                "後端已有這個策略的錯誤草稿。先 get_strategy 並 validate_strategy，"
                "必須先取得失敗診斷，再修改儲存，不能直接覆寫而跳過診斷。"
            )
        else:
            preparation = "請先查詢策略清單。"
        return f"""請透過 TradingDev MCP 開發策略 {self.strategy_id}。
{preparation}
讀取 get_strategy_contract，依照契約完成 Python 與 YAML。{self.requirement}
所有參數放在 strategy.parameters 且可覆寫：{self.parameters}。
YAML backtest 設定 BTC/USDT、1h、2024-01-01 至 2024-01-08、init_cash=10000、
mode=signal、fees=0、slippage=0、random_seed=42。data.requirements.features 為空。
完成 save_strategy、validate_strategy、dry_run_strategy；若有錯誤請讀取診斷修正，
每次 save 取得的新 revision_id 必須傳給 validate、dry-run 與 start_backtest，
確認各工具回覆的 revision_id 一致，直到 runnable。不 promote。
接著你必須親自透過 MCP start_backtest 啟動這個策略，
symbol=BTC/USDT、timeframe=1h、start_date=2024-01-01、end_date=2024-01-08。
本次行情已預先放入後端快取，不下載行情、不使用外部資料。
持續 get_job_status 查詢直到 done，再以回傳的 run_id 呼叫 get_run，
以及 list_artifacts，確認結果後就結束，只回答「完成」與 run_id。
不得只啟動工作就結束；不需要額外取得 artifact 內容或撰寫回測報告。
只能透過 MCP 工具撰寫與執行，不使用 shell 或直接編輯檔案。
直接呼叫工具，不敘述計畫或重貼程式碼；程式、YAML 與 request_summary 保持精簡，
但不得省略需求或驗證步驟。
"""


_SMA_REQUIREMENT = (
    "用收盤價 5 根與 20 根簡單均線判斷：短均線>長均線時每根 signal=1，"
    "小於時每根=-1，相等或資料不足20根時=0。不得修改輸入資料或使用未來資料。"
)
SCENARIOS = {
    "sma": Scenario(
        "sma",
        {"fast_period": 5, "slow_period": 20},
        {"fast_period": 3, "slow_period": 8},
        _SMA_REQUIREMENT,
    ),
    "momentum": Scenario(
        "momentum",
        {"lookback": 6},
        {"lookback": 3},
        "比較每根 close 與 lookback 根前的 close；較高每根 signal=1，較低=-1，"
        "相等或沒有足夠歷史=0。不得修改輸入資料或使用未來資料。",
    ),
    "repair": Scenario(
        "repair",
        {"fast_period": 5, "slow_period": 20},
        {"fast_period": 3, "slow_period": 8},
        _SMA_REQUIREMENT,
        repair=True,
    ),
    "legacy": Scenario(
        "legacy",
        {"fast_period": 5, "slow_period": 20},
        {"fast_period": 3, "slow_period": 8},
        _SMA_REQUIREMENT,
        legacy=True,
    ),
}


def market_frame() -> pd.DataFrame:
    close = pd.Series(
        [
            100.0 + value
            for value in ([*range(1, 41), *range(40, 0, -1), *([1] * 16)] * 2)
        ]
    )
    return pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2024-01-01", periods=len(close), freq="h", tz="UTC"
            ),
            "open": close,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": 100.0,
        }
    )


def expected_signals(
    frame: pd.DataFrame, scenario: Scenario, parameters: dict[str, int]
) -> pd.Series:
    close = frame["close"]
    if scenario.name == "momentum":
        left, right = close, close.shift(parameters["lookback"])
    else:
        left = close.rolling(parameters["fast_period"]).mean()
        right = close.rolling(parameters["slow_period"]).mean()
    expected = pd.Series(0, index=frame.index, name="signal")
    expected.loc[left > right] = 1
    expected.loc[left < right] = -1
    return expected
