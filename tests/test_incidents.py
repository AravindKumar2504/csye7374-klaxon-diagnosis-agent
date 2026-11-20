"""The real incident data, captured from Klaxon's live stack (incidents/), and the offline demo on it."""

from __future__ import annotations

import json

import pytest

from diagnosis_agent import KLAXON_COMMIT, cases, check, loop, scripted
from diagnosis_agent.tools import DATA_FILES, IncidentTools

NAMES = ("duplicate-delivery", "poison-message")


@pytest.mark.parametrize("name", NAMES)
def test_each_incident_was_captured_from_klaxon_at_the_copied_commit(name):
    folder = cases.INCIDENTS / name
    for file in (*DATA_FILES, "harness.json"):
        assert (folder / file).is_file(), file
    about = json.loads((folder / "incident.json").read_text())
    assert about["name"] == name
    assert about["captured"]["klaxon_commit"] == KLAXON_COMMIT and about["captured"]["klaxon_code_clean"] is True
    assert about["label"].startswith("Captured from Klaxon's synthetic payments system")
    assert about["log_lines"] == sum(1 for _ in open(folder / "logs.jsonl"))


@pytest.mark.parametrize("name", NAMES)
def test_the_alert_was_raised_by_klaxons_rules_before_the_capture(name):
    about = json.loads((cases.INCIDENTS / name / "incident.json").read_text())
    alert = loop.Alert(**about["alert"])  # valid as the loop takes alerts: one line each, UTC
    assert alert.fired_at <= about["money_check_called_at"] <= about["logs_read_at"]
    harness = json.loads((cases.INCIDENTS / name / "harness.json").read_text())["record"]
    assert harness["alert"] == about["alert"] and harness["cleared"] is not None


@pytest.mark.parametrize("name", NAMES)
def test_every_tool_answers_on_the_real_data(name):
    tools = IncidentTools(cases.INCIDENTS / name)
    tools.open()
    for tool, arguments in (
        ("money_check", {}),
        ("recent_deploys", {}),
        ("search_logs", {"service": "ledger-consumer"}),
    ):
        answer = tools.call(tool, arguments, 5.0)
        assert not answer.is_error and len(answer.text) < 60_000
        json.loads(answer.text)


def test_the_duplicate_incident_shows_its_cause():
    folder = cases.INCIDENTS / "duplicate-delivery"
    money = json.loads((folder / "money_check.json").read_text())
    failed = {inv["name"] for inv in money["invariants"] if not inv["passed"]}
    assert "posted_exactly_once" in failed
    deploys = json.loads((folder / "recent_deploys.json").read_text())["deploys"]
    assert deploys[0]["service"] == "ledger-consumer" and deploys[0]["settings"]["LEDGER_HANDLER"] == "naive"


def test_the_poison_incident_shows_its_cause():
    folder = cases.INCIDENTS / "poison-message"
    money = json.loads((folder / "money_check.json").read_text())
    assert money["outcome"] == "could not run"
    deploys = json.loads((folder / "recent_deploys.json").read_text())["deploys"]
    assert deploys[0]["service"] == "ledger-consumer" and deploys[0]["settings"]["BAD_EVENTS"] == "retry"


@pytest.mark.parametrize("name", list(cases.CASES))
def test_the_offline_demo_passes_every_check_on_the_real_data(tmp_path, name):
    case = cases.get(name)
    folder = tmp_path / name
    folder.mkdir()
    (folder / check.RUN_FILE).write_text(json.dumps({"case": name}))
    result = loop.investigate_offline(case.alert(), cases.BUDGETS, folder, {}, scripted.model_for(case), case.tools())
    assert result.reason == "diagnosed"
    checked = check.check_run(folder)
    assert checked["passed"], [item for item in checked["checks"] if not item["passed"]]
