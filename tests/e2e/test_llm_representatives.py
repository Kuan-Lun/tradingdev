"""Natural multi-turn selection, pinned reads, explicit revision and recovery."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import pytest

from tests.e2e.llm_client import ToolCall
from tests.e2e.test_llm_comparison import _read_only_tools, _run
from tests.integration.mcp_harness import temporary_mcp_workspace
from tests.representative_fixtures import publish_representative_run
from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.adapters.storage.sqlite import SQLiteStore
from tradingdev.app.representative_service import RepresentativeService

if TYPE_CHECKING:
    from tests.e2e.conversation import ModelTurn
    from tests.integration.mcp_harness import MCPWorkspace

REQUEST = (
    "工作區已保存回測 representative_A。請把它的完整回測結果固定為『BTC 研究』"
    "中這個策略的代表，之後新增實驗也保留這份選擇。這次只保存選擇，"
    "不要執行回測或修改策略。完成後告訴我保存的代表與版本。"
)


def _check_attempts(
    calls: list[ToolCall], read_only: set[str], allowed_writes: set[str]
) -> None:
    assert not [
        call.name for call in calls if call.name not in read_only | allowed_writes
    ], "Representative workflows must inspect every attempt, including failed writes"


def test_readonly_representative_request_rejects_failed_reselection_attempts() -> None:
    calls = [ToolCall("get_strategy_representatives", {}, {"success": True})]
    _check_attempts(calls, {"get_strategy_representatives"}, set())
    for name in (
        "revise_strategy_representative",
        "create_strategy_representative",
        "prepare_backtest",
        "unknown_tool",
    ):
        with pytest.raises(AssertionError, match="every attempt"):
            _check_attempts(
                [*calls, ToolCall(name, {}, {"success": False})],
                {"get_strategy_representatives"},
                set(),
            )


class RepresentativeUser:
    def __init__(self, workspace: MCPWorkspace, read_only: set[str]) -> None:
        self.workspace = workspace
        self.paths = WorkspacePaths(workspace.workspace)
        self.store = SQLiteStore(self.paths)
        self.service = RepresentativeService(self.paths, self.store)
        self.read_only = read_only
        self.identity = ""
        self.turns: list[ModelTurn] = []

    def assert_ranking(
        self, turn: ModelTurn, version: int, run_id: str, *, unavailable: bool = False
    ) -> None:
        rankings = [call for call in turn.calls if call.name == "rank_candidates"]
        assert rankings, (
            "A saved representative must be ranked through the shared surface"
        )
        for call in rankings:
            assert call.arguments.get("representatives") == [
                {"representative_id": self.identity, "version": version}
            ]
            assert call.arguments.get("metric") == "run.sharpe_ratio"
            assert call.arguments.get("runs") is None
            assert call.result.get("success") is True
            if unavailable:
                assert call.result["n_ranked"] == 0 and call.result["n_skipped"] == 1
                assert call.result["skipped"][0]["source_available"] is False
            else:
                assert [
                    (r["run_id"], r["representative"]["version"])
                    for r in call.result["ranked"]
                ] == [(run_id, version)]

    async def __call__(self, turn: ModelTurn) -> str | None:
        self.turns.append(turn)
        number = len(self.turns)
        allowed = (
            {"create_strategy_representative"}
            if number == 1
            else {"revise_strategy_representative"}
            if number in (3, 4)
            else set()
        )
        _check_attempts(turn.calls, self.read_only, allowed)
        print(f"[representative turn {number}]\n{turn.text}", flush=True)
        if number == 1:
            page = self.service.get(group_id="BTC 研究")
            assert page["total"] == 1
            self.identity = page["items"][0]["representative_id"]
            assert page["items"][0]["version"] == 1
            assert page["items"][0]["run_id"] == "representative_A"
            publish_representative_run(
                self.paths,
                self.store,
                "representative_B",
                window=20,
                revision="revision-2",
                value=9.0,
            )
            return (
                "工作區現在又新增了分數更高的 representative_B。請用已保存的代表"
                "按 Sharpe 排名，並說明現在的代表是哪次結果。這次只查詢，不要改選。"
            )
        if number == 2:
            self.assert_ranking(turn, 1, "representative_A")
            assert self.service.get(self.identity)["representative"]["version"] == 1
            return (
                "現在我明確將『BTC 研究』中這個策略的代表改選為 "
                "representative_B 的完整結果。"
                "請以剛才第一版作為預期版本修訂，完成後用新代表排名，並查回第一版的原始選擇。"
            )
        if number == 3:
            self.assert_ranking(turn, 2, "representative_B")
            assert self.service.get(self.identity)["representative"]["version"] == 2
            assert any(
                call.name == "get_strategy_representatives"
                and call.arguments.get("version") == 1
                and call.result.get("representative", {})
                .get("snapshot", {})
                .get("run_id")
                == "representative_A"
                for call in turn.calls
            )
            return (
                "另一份編輯仍停在第一版。我想嘗試以第一版作為預期版本改回 "
                "representative_A 的完整結果。"
                "請實際提交這一次舊版本修改；如果衝突就保留目前代表並告訴我衝突，"
                "不要改用較新版強行重試。"
            )
        if number == 4:
            changes = [
                c for c in turn.calls if c.name == "revise_strategy_representative"
            ]
            assert len(changes) == 1
            assert changes[0].arguments["expected_version"] == 1
            assert changes[0].result.get("code") == "representative_conflict"
            assert self.service.get(self.identity)["representative"]["version"] == 2
            (self.paths.runs / "representative_A" / "performance.json").unlink()
            return (
                "第一版代表的原始來源 representative_A 現在遺失了。"
                "請查詢並排名歷史第一版，"
                "說明不可用原因；不要重跑，也不要用第二版或較高分的新實驗取代第一版。"
            )
        assert number == 5
        self.assert_ranking(turn, 1, "representative_A", unavailable=True)
        assert self.service.get(self.identity)["representative"]["version"] == 2
        assert not self.store.list_jobs(), (
            "Saved-selection workflows must never execute"
        )
        return None


@pytest.mark.live_llm
def test_natural_representative_selection_and_explicit_revision(
    pytestconfig: pytest.Config,
) -> None:
    with temporary_mcp_workspace() as workspace:
        paths = WorkspacePaths(workspace.workspace)
        publish_representative_run(
            paths, SQLiteStore(paths), "representative_A", value=0.5
        )

        async def run() -> RepresentativeUser:
            async with workspace.connect() as client:
                read_only = await _read_only_tools(client)
            user = RepresentativeUser(workspace, read_only)
            calls: list[ToolCall] = []
            try:
                await _run(workspace, pytestconfig, user, calls, REQUEST)
            except BaseException as exc:
                for turn in user.turns:
                    exc.add_note(f"Representative turn {turn.number}:\n{turn.text}")
                exc.add_note(
                    f"All tool attempts: {[(c.name, c.arguments) for c in calls]}"
                )
                raise
            assert len(user.turns) == 5
            return user

        asyncio.run(run())
