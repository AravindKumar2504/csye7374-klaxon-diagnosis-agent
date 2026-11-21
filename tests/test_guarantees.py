"""The loop's safety guarantees, one test each, written so that the test fails if its guarantee is taken out.

A code review on 2026-11-21 UTC took nine guarantees out of the code one at a time and found that the tests still all
passed. Each test below covers one of the nine. Also here: the loop's other stop paths that no test reached.
"""

from __future__ import annotations

import json
import shutil

import pytest

from diagnosis_agent import REPO, cases, loop
from diagnosis_agent.model import ScriptedModel
from diagnosis_agent.tool_client import ToolError, ToolResult, ToolSpec
from diagnosis_agent.trace import Trace, read_trace

LIVE_RUN = REPO / "runs" / "2026-11-20T091900Z_poison-message_claude-opus-5-5"


def call(tool, arguments=None):
    return json.dumps({"action": "call_tool", "why": "test", "tool": tool, "arguments": arguments or {}})


def diagnosis(evidence=("c1",)):
    return json.dumps(
        {
            "action": "diagnose",
            "why": "test",
            "diagnosis": {
                "hypothesis": "The ledger consumer posts redelivered webhooks twice: its dedupe is gone.",
                "evidence": [{"call_id": c, "shows": "it shows the symptom"} for c in evidence],
                "confidence": 0.5,
                "proposed_fix": {"kind": "rollback", "target": "ledger-consumer", "summary": "roll it back"},
                "blast_radius": "the ledger consumer",
                "rollback_plan": "deploy it again",
            },
        }
    )


def run(tmp_path, incidents, model, budgets=cases.BUDGETS, clock=None, tools=None, case_name="duplicate-delivery"):
    case = cases.get(case_name)
    kwargs = {} if clock is None else {"clock": clock}
    return loop.investigate(
        case.alert(incidents), model, tools or case.tools(incidents), tmp_path / "trace.jsonl", budgets, **kwargs
    )


# 1. Line breaks: a tool's result reaches the model on one line, whatever line separators it holds.


def test_a_result_with_unicode_line_separators_stays_on_its_one_line():
    message = "one two three"
    text = json.dumps({"lines": [{"message": message}]}, ensure_ascii=False)
    turn = loop.observation("c1", "search_logs", {}, ToolResult(text, False))
    lines = turn.splitlines()  # splits on U+2028 and U+2029 too, as many readers do
    assert len(lines) == 3 and lines[1] == loop.UNTRUSTED
    assert " " not in turn and " " not in turn
    assert json.loads(lines[2])["result"]["lines"][0]["message"] == message  # escaped, not lost


# 2. Exact replay: a replay that differs from the recorded run says so.


def test_a_replay_that_differs_from_its_recording_is_not_reproduced(tmp_path):
    folder = shutil.copytree(LIVE_RUN, tmp_path / LIVE_RUN.name)
    _, _, same = loop.replay(folder, tmp_path / "again.jsonl")
    assert same == []
    records = [json.loads(line) for line in (folder / loop.TRACE).read_text().splitlines()]
    records[1]["why"] = "a reason the loop never gave"
    (folder / loop.TRACE).write_text("".join(json.dumps(record) + "\n" for record in records))
    _, _, differences = loop.replay(folder, tmp_path / "again-2.jsonl")
    assert differences == ["step 1 differs in: why"]


def test_compare_finds_a_changed_reply_a_missing_step_and_another_stop():
    recorded = read_trace(LIVE_RUN / loop.TRACE)
    assert loop.compare(recorded, recorded) == []
    changed = json.loads(json.dumps(recorded))
    changed[2]["reply"]["sha256"] = "0" * 64
    assert loop.compare(recorded, changed) == ["step 2 differs in: reply"]
    shorter = recorded[:3] + recorded[-1:]
    assert any("took 8 steps, the replay 2" in line for line in loop.compare(recorded, shorter))
    stopped = json.loads(json.dumps(recorded))
    stopped[-1]["reason"] = "step_budget"
    assert "the recorded run stopped on diagnosed, the replay on step_budget" in loop.compare(recorded, stopped)


# 3. The allow-list, checked again at the point of use.


def test_the_point_of_use_check_refuses_what_this_run_may_not_call(tmp_path, incidents):
    case = cases.get("duplicate-delivery")
    tools = case.tools(incidents)
    tools.open()
    with Trace(tmp_path / "trace.jsonl") as trace:
        investigation = loop.Investigation(
            case.alert(incidents), ScriptedModel([]), tools, trace, cases.BUDGETS, lambda: 0.0, {}
        )
        investigation.schemas = {"money_check": {}}  # what this run was offered
        with pytest.raises(loop.NotAllowed):
            investigation._call("rollback_deploy", {}, 1.0)  # not a read-only tool
        with pytest.raises(loop.NotAllowed):
            investigation._call("search_logs", {}, 1.0)  # read-only, but not offered to this run
        assert investigation._call("money_check", {}, 1.0).is_error is False


# 4. The read-only marking: a tool on the allow-list must also be marked read-only and not destructive.


@pytest.mark.parametrize(
    "annotations",
    [{"readOnlyHint": False}, {"readOnlyHint": True, "destructiveHint": True}, {}],
    ids=["not read-only", "destructive", "unmarked"],
)
def test_a_tool_on_the_allow_list_but_not_marked_read_only_is_refused(annotations):
    spec = ToolSpec("money_check", "Money check", "reads the books", {"type": "object"}, annotations)
    with pytest.raises(loop.ToolsRefused, match="read-only"):
        loop.allowed_tools([spec])


# 5 is the read_only evidence check: tests/test_check.py, test_the_read_only_check_fails_*.

# 6. The run's time budget.


class Clock:
    """A clock that moves only when a test moves it."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def test_the_time_budget_stops_the_run_before_the_next_step(tmp_path, incidents):
    clock = Clock()

    def slow_reply(text):
        def reply(system, turns):
            clock.now += 400.0  # each reply takes 400 s of a 600 s budget
            return text

        return reply

    model = ScriptedModel([slow_reply(call("money_check")), slow_reply(call("search_logs")), diagnosis()])
    result = run(tmp_path, incidents, model, clock=clock)
    assert result.reason == "time_budget" and "all 600 s of the run are used" in result.detail
    assert len(model.calls) == 2 and result.tool_calls == 1  # the second call was refused: no time was left


# 7. The run's step budget, checked before each step, whatever used the steps up.


def test_the_step_budget_stops_the_run_when_invalid_replies_used_the_steps(tmp_path, incidents):
    budgets = loop.Budgets(steps=2, tool_calls=8, seconds=600.0)
    model = ScriptedModel(["{", "{", "{"])
    result = run(tmp_path, incidents, model, budgets)
    assert result.reason == "step_budget" and "all 2 steps of the run are used" in result.detail
    assert len(model.calls) == 2


def test_a_tool_call_on_the_last_step_is_refused(tmp_path, incidents):
    budgets = loop.Budgets(steps=2, tool_calls=8, seconds=600.0)  # 2 is the smallest step budget a run may have
    result = run(tmp_path, incidents, ScriptedModel([call("money_check"), call("search_logs")]), budgets)
    assert result.reason == "step_budget" and result.tool_calls == 1
    assert "no step is left to read the result" in result.detail


# 8. A result too long for the model is cut, and the model is told.


def test_a_result_over_60000_characters_is_cut_and_the_model_told():
    turn = loop.observation("c1", "search_logs", {}, ToolResult("x" * 70_000, False))
    assert loop.MAX_OBSERVATION == 60_000
    assert "The result was cut to its first 60,000 characters" in turn
    assert len(turn) < 61_000


# 9. A live adapter's own deadline is lowered to the run's time left for each call, then put back.


class TimedModel(ScriptedModel):
    """A scripted model with a live adapter's deadline: it notes the deadline each call is given."""

    def __init__(self, script, clock):
        super().__init__(script)
        self.timeout_seconds = 300.0
        self.given: list[float] = []
        self.clock = clock

    def ask(self, system, messages):
        self.given.append(self.timeout_seconds)
        self.clock.now += 400.0
        return super().ask(system, messages)


def test_the_live_deadline_is_lowered_to_the_time_left_and_put_back(tmp_path, incidents):
    clock = Clock()
    model = TimedModel([call("money_check"), diagnosis()], clock)
    run(tmp_path, incidents, model, clock=clock)
    assert model.given == [300.0, 200.0]  # 600 s left, then 200 s left
    assert model.timeout_seconds == 300.0


# The loop's other stop paths.


class BrokenTools:
    """The incident's tools, except that money_check fails as a tool server can: no result at all."""

    def __init__(self, inner):
        self.inner = inner

    def open(self, timeout=30.0):
        return self.inner.open(timeout)

    def call(self, name, arguments, timeout):
        if name == "money_check":
            raise ToolError("the tool server closed its pipe")
        return self.inner.call(name, arguments, timeout)

    def close(self):
        self.inner.close()


def test_a_tool_that_fails_mid_run_stops_it_and_both_records_say_so(tmp_path, incidents):
    case = cases.get("duplicate-delivery")
    folder = tmp_path / "run"
    folder.mkdir()
    model = ScriptedModel([call("search_logs", {"limit": 1}), call("money_check"), diagnosis()])
    result = loop.investigate_offline(
        case.alert(incidents), cases.BUDGETS, folder, {}, model, BrokenTools(case.tools(incidents))
    )
    assert result.reason == "tools_failed" and "closed its pipe" in result.detail
    steps = [record for record in read_trace(folder / loop.TRACE) if record["kind"] == "step"]
    assert steps[-1]["ran"]["tool"] == "money_check" and "it gave no result (ToolError)" in steps[-1]["why"]
    recorded = [json.loads(line) for line in (folder / loop.TOOL_RESULTS).read_text().splitlines()]
    assert recorded[-1]["outcome"] == "failed" and "closed its pipe" in recorded[-1]["error"]
    assert len(model.calls) == 2  # the run stopped: the model was not asked again
