"""The three tools, on the hand-written fixture incidents (conftest.py)."""

from __future__ import annotations

import json

import pytest

from diagnosis_agent import cases, loop
from diagnosis_agent.tools import DOCKER_SILENT, SIMULATED, IncidentDataError, IncidentTools, Withheld


def opened(folder):
    tools = IncidentTools(folder)
    specs = tools.open()
    return tools, specs


def result(tools, name, arguments=None):
    answer = tools.call(name, arguments or {}, 5.0)
    return answer, (None if answer.is_error else json.loads(answer.text))


def test_three_tools_all_read_only_and_on_the_allow_list(incidents):
    _, specs = opened(incidents / "duplicate-delivery")
    assert [spec.name for spec in specs] == ["search_logs", "recent_deploys", "money_check"]
    assert all(spec.read_only for spec in specs)
    assert loop.allowed_tools(specs) == specs


def test_money_check_and_deploys_return_the_saved_results(incidents):
    folder = incidents / "duplicate-delivery"
    tools, _ = opened(folder)
    _, money = result(tools, "money_check")
    assert money == json.loads((folder / "money_check.json").read_text())
    _, deploys = result(tools, "recent_deploys")
    assert deploys["deploys"][0]["settings"]["LEDGER_HANDLER"] == "naive"


def test_search_logs_filters_newest_first(incidents):
    tools, _ = opened(incidents / "duplicate-delivery")
    _, found = result(tools, "search_logs", {"contains": "evt_000101"})
    assert found["matched"] == 5
    assert [line["time"] for line in found["lines"]] == sorted((line["time"] for line in found["lines"]), reverse=True)
    _, posted = result(tools, "search_logs", {"service": "ledger-consumer", "contains": "EVT_000101"})
    assert posted["matched"] == 2 and all(line["message"] == "posted" for line in posted["lines"])
    _, warned = result(tools, "search_logs", {"level": "WARNING"})
    assert warned["matched"] == 1


def test_search_logs_minutes_and_limit(incidents):
    tools, _ = opened(incidents / "duplicate-delivery")
    _, recent = result(tools, "search_logs", {"service": "payments-api"})  # 15 minutes by default
    assert recent["matched"] == 1  # the 06:20 line is 41 minutes before the data's end
    _, older = result(tools, "search_logs", {"service": "payments-api", "minutes": 60})
    assert older["matched"] == 2
    _, capped = result(tools, "search_logs", {"limit": 2})
    assert len(capped["lines"]) == 2 and capped["more_may_exist"] is True


@pytest.mark.parametrize(
    "arguments, said",
    [
        ({"service": "ledger"}, "service must be one of"),
        ({"level": "WARN"}, "level must be one of"),
        ({"limit": 500}, "limit must be from 1 to 50"),
        ({"query": '{service_name="x"}'}, "unknown argument"),
        ({"contains": ""}, "must not be empty"),
    ],
)
def test_bad_arguments_come_back_as_tool_errors(incidents, arguments, said):
    tools, _ = opened(incidents / "duplicate-delivery")
    answer, _ = result(tools, "search_logs", arguments)
    assert answer.is_error and said in answer.text


def test_no_arguments_tools_refuse_arguments(incidents):
    tools, _ = opened(incidents / "duplicate-delivery")
    answer, _ = result(tools, "money_check", {"wait": False})
    assert answer.is_error and "unknown argument" in answer.text


def test_withheld_deploys_answer_the_docker_error_marked_simulated(incidents):
    tools = Withheld(IncidentTools(incidents / "duplicate-delivery"), {"recent_deploys": DOCKER_SILENT})
    tools.open()
    answer = tools.call("recent_deploys", {}, 5.0)
    assert answer.is_error and answer.text == DOCKER_SILENT and answer.simulated_by == SIMULATED
    assert not tools.call("money_check", {}, 5.0).is_error  # the other tools still answer
    assert tools.server_info["withheld"] == {"recent_deploys": DOCKER_SILENT}


def test_the_missing_evidence_case_withholds_only_the_deploy_list(incidents):
    case = cases.get("missing-deploys")
    tools = case.tools(incidents)
    tools.open()
    assert tools.call("recent_deploys", {}, 5.0).is_error
    assert not tools.call("search_logs", {}, 5.0).is_error


def test_server_info_records_the_data_hashes(incidents):
    tools, _ = opened(incidents / "duplicate-delivery")
    info = tools.server_info
    assert info["incident"] == "duplicate-delivery"
    assert set(info["data_sha256"]) == {"incident.json", "money_check.json", "recent_deploys.json", "logs.jsonl"}


def test_damaged_data_is_refused(incidents):
    folder = incidents / "duplicate-delivery"
    (folder / "logs.jsonl").write_text('{"time": 5}\n')
    with pytest.raises(IncidentDataError):
        IncidentTools(folder).open()


def test_unreadable_data_is_refused_with_its_folder_named(incidents):
    folder = incidents / "duplicate-delivery"
    (folder / "money_check.json").write_text("{not json")
    with pytest.raises(IncidentDataError, match=r"duplicate-delivery cannot be read \(JSONDecodeError\)"):
        IncidentTools(folder).open()


def test_search_logs_matches_text_in_a_message_too(incidents):
    tools, _ = opened(incidents / "duplicate-delivery")
    _, body = result(tools, "search_logs", {"contains": "previous instructions"})
    assert body["matched"] == 1 and body["lines"][0]["level"] == "WARNING"  # the words are only in its message


def test_an_unknown_tool_and_arguments_that_are_not_an_object_are_tool_errors(incidents):
    tools, _ = opened(incidents / "duplicate-delivery")
    answer, _ = result(tools, "rollback_deploy")
    assert answer.is_error and answer.text == "Unknown tool: rollback_deploy"
    for arguments in ([["service", "ledger-consumer"]], "evt_000101", None, 7):
        answer = tools.call("search_logs", arguments, 5.0)
        assert answer.is_error and answer.text == "arguments must be a JSON object"


def test_a_result_too_large_is_refused_with_how_to_ask_for_less(incidents, monkeypatch):
    from diagnosis_agent import tools as module

    monkeypatch.setattr(module, "MAX_RESULT", 200)
    tools, _ = opened(incidents / "duplicate-delivery")
    answer, _ = result(tools, "search_logs")
    assert answer.is_error and answer.text.startswith("the result was too large") and "a lower limit" in answer.text


def test_a_bug_in_a_tool_is_a_tool_error_the_model_can_read(incidents, monkeypatch):
    from diagnosis_agent import tools as module

    def broken(incident, arguments):
        raise KeyError("time")

    monkeypatch.setattr(module, "search_logs", broken)
    tools, _ = opened(incidents / "duplicate-delivery")
    answer, _ = result(tools, "search_logs")
    assert answer.is_error and answer.text == "the tool failed unexpectedly (KeyError)"
