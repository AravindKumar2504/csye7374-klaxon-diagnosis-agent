"""The command line's other paths, run in this process: a failing run, an interrupted one, check --write, replays
that do not reproduce, folders that are not runs, bad tool arguments, and show."""

from __future__ import annotations

import json
import shutil

import pytest

from diagnosis_agent import REPO, check, cli, loop, scripted
from diagnosis_agent.model import ScriptedModel

POISON = REPO / "runs" / "2026-11-20T091900Z_poison-message_claude-opus-5-5"


def call(tool, arguments=None):
    return json.dumps({"action": "call_tool", "why": "test", "tool": tool, "arguments": arguments or {}})


def made_up_diagnosis():
    return json.dumps(
        {
            "action": "diagnose",
            "why": "test",
            "diagnosis": {
                "hypothesis": "The ledger consumer books evt_999999 twice: its dedupe is gone.",
                "evidence": [{"call_id": "c1", "shows": "evt_999999 posted 2 times"}],
                "confidence": 0.5,
                "proposed_fix": {"kind": "rollback", "target": "ledger-consumer", "summary": "roll it back"},
                "blast_radius": "the ledger consumer",
                "rollback_plan": "deploy it again",
            },
        }
    )


def test_a_run_whose_diagnosis_fails_a_check_exits_1_and_prints_its_steps(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(scripted, "model_for", lambda case: ScriptedModel([call("money_check"), made_up_diagnosis()]))
    assert cli.main(["run", "duplicate-delivery", "--model", "scripted", "--out", str(tmp_path)]) == 1
    out = capsys.readouterr().out
    assert "Step 1: money_check {}" in out and "back: c1, failed: 2 of 5 invariants" in out
    assert "Blast radius: the ledger consumer" in out and "Rollback plan: deploy it again" in out
    assert "version 2 (written after them): FAILED a check" in out and "evt_999999" in out
    (folder,) = list(tmp_path.iterdir())
    assert not json.loads((folder / check.CHECK_FILE).read_text())["passed"]


def test_an_interrupted_run_exits_130_and_says_where_its_trace_is(tmp_path, monkeypatch, capsys):
    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(loop, "investigate_offline", interrupted)
    assert cli.main(["run", "poison-message", "--model", "scripted", "--out", str(tmp_path)]) == 130
    assert "Interrupted; the trace says so" in capsys.readouterr().err


def test_check_write_saves_each_version_beside_the_run_as_committed(tmp_path, capsys):
    folder = shutil.copytree(POISON, tmp_path / POISON.name)
    for name in check.CHECK_FILES.values():
        (folder / name).unlink()
    assert cli.main(["check", str(folder), "--write"]) == 0
    assert "version 2 (written after them): PASSED every check" in capsys.readouterr().out
    assert cli.main(["check", str(folder), "--version", "1", "--write"]) == 0
    for name in check.CHECK_FILES.values():
        assert (folder / name).read_bytes() == (POISON / name).read_bytes()


def test_check_and_show_refuse_a_folder_that_is_not_a_run(capsys):
    incident = REPO / "incidents" / "duplicate-delivery"
    assert cli.main(["check", str(incident)]) == 2
    assert cli.main(["show", str(incident)]) == 2
    assert capsys.readouterr().err.count("Not a run folder") == 2


def test_replay_says_when_a_run_does_not_reproduce_or_cannot_be_replayed(tmp_path, capsys):
    folder = shutil.copytree(POISON, tmp_path / POISON.name)
    trace = folder / loop.TRACE
    records = [json.loads(line) for line in trace.read_text().splitlines()]
    records[1]["proposed"]["why"] = "a reason the model never gave"
    trace.write_text("".join(json.dumps(record) + "\n" for record in records))
    assert cli.main(["replay", str(folder)]) == 1
    assert "Not reproduced:\n  step 1 differs in: proposed" in capsys.readouterr().out
    (folder / loop.RECORDING).unlink()
    assert cli.main(["replay", str(folder)]) == 2
    assert "Cannot replay" in capsys.readouterr().err


@pytest.mark.parametrize("arguments", ["{not json", '"evt_000308"', "null", "[1]", '[["service", "ledger-consumer"]]'])
def test_tool_refuses_arguments_that_are_not_a_json_object(capsys, arguments):
    assert cli.main(["tool", "poison-message", "search_logs", arguments]) == 1
    assert capsys.readouterr().out.startswith("TOOL ERROR: ")


def test_show_prints_each_step_with_what_came_back_and_both_versions_of_the_checks(capsys):
    assert cli.main(["show", str(POISON)]) == 0
    out = capsys.readouterr().out
    assert "Alert: BooksBehindProvider, fired at 2026-11-20T07:56:59Z" in out
    assert "Step 1: money_check {}\n  why:  Confirm the symptom first" in out
    assert "back: c1, could not run: not settled up to the cutoff" in out
    assert (
        "back: c3, 10 deploys; newest: ledger-consumer at 07:56:08, LEDGER_HANDLER=idempotent, BAD_EVENTS=retry" in out
    )
    assert "Step 7: diagnose" in out and "refused: invalid reply: diagnosis.hypothesis: must be at most 1000" in out
    assert "Proposed fix (not applied): config_change ledger-consumer" in out
    assert "version 1 (committed before the live runs): PASSED every check" in out
    assert "version 2 (written after them): PASSED every check" in out
