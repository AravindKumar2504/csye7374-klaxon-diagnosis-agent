"""The evidence checks: a right diagnosis passes; each kind of claim the evidence does not hold fails its check."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from diagnosis_agent import cases, check, loop, scripted
from diagnosis_agent.model import ScriptedModel


def run_and_check(tmp_path, incidents, case_name, script):
    case = cases.get(case_name)
    folder = tmp_path / case_name
    folder.mkdir()
    (folder / check.RUN_FILE).write_text(json.dumps({"case": case_name}))
    model = script if isinstance(script, ScriptedModel) else ScriptedModel(script)
    loop.investigate_offline(case.alert(incidents), cases.BUDGETS, folder, {}, model, case.tools(incidents))
    return check.check_run(folder, incidents)


def by_name(result):
    return {item["check"]: item for item in result["checks"]}


def call(tool, arguments=None):
    return json.dumps({"action": "call_tool", "why": "test", "tool": tool, "arguments": arguments or {}})


def diagnose(hypothesis, evidence, fix_summary="Roll the ledger consumer back."):
    return json.dumps(
        {
            "action": "diagnose",
            "why": "test",
            "diagnosis": {
                "hypothesis": hypothesis,
                "evidence": [{"call_id": c, "shows": s} for c, s in evidence],
                "confidence": 0.7,
                "proposed_fix": {"kind": "rollback", "target": "ledger-consumer", "summary": fix_summary},
                "blast_radius": "the ledger consumer",
                "rollback_plan": "deploy it again",
            },
        }
    )


RIGHT = (
    "The ledger consumer books provider events twice: the provider redelivers evt_000101 and the ledger consumer posts"
    " it again because its dedupe is gone."
)


@pytest.mark.parametrize("name", ["duplicate-delivery", "poison-message", "missing-deploys"])
def test_the_scripted_runs_pass_every_check(tmp_path, incidents, name):
    result = run_and_check(tmp_path, incidents, name, scripted.model_for(cases.get(name)))
    assert result["passed"], [item for item in result["checks"] if not item["passed"]]
    assert result["path"] == ["money_check", "search_logs", "recent_deploys"]


def test_a_right_diagnosis_with_right_citations_passes(tmp_path, incidents):
    script = [
        call("money_check"),
        call("search_logs", {"contains": "evt_000101"}),
        diagnose(
            RIGHT, [("c1", "evt_000101 posted 2 times, $12.34"), ("c2", "evt_000101 queued twice and posted twice")]
        ),
    ]
    result = run_and_check(tmp_path, incidents, "duplicate-delivery", script)
    assert result["passed"]
    assert by_name(result)["names_the_cause"]["detail"] == "the rule says: names the cause"


def test_a_fact_in_the_evidence_that_its_cited_result_lacks_fails(tmp_path, incidents):
    script = [
        call("money_check"),
        call("search_logs", {"contains": "evt_000101"}),
        # c2 holds evt_000101, but no pay_ id: that one is only in c1's result.
        diagnose(RIGHT, [("c1", "evt_000101 posted 2 times"), ("c2", "pay_000090 was charged twice")]),
    ]
    result = run_and_check(tmp_path, incidents, "duplicate-delivery", script)
    item = by_name(result)["cited_results_hold_the_facts"]
    assert not item["passed"] and "pay_000090" in item["detail"]
    assert not result["passed"]


def test_an_invented_id_in_the_hypothesis_fails(tmp_path, incidents):
    script = [
        call("money_check"),
        diagnose(RIGHT.replace("evt_000101", "evt_009999"), [("c1", "the money check failed")]),
    ]
    result = run_and_check(tmp_path, incidents, "duplicate-delivery", script)
    item = by_name(result)["hypothesis_facts_in_results"]
    assert not item["passed"] and "evt_009999" in item["detail"]


def test_a_wrong_amount_fails_and_a_right_one_in_cents_passes(tmp_path, incidents):
    wrong = run_and_check(
        tmp_path,
        incidents,
        "duplicate-delivery",
        [call("money_check"), diagnose(RIGHT + " It is $99.99.", [("c1", "x")])],
    )
    assert not by_name(wrong)["hypothesis_facts_in_results"]["passed"]
    (tmp_path / "duplicate-delivery").rename(tmp_path / "first")
    right = run_and_check(
        tmp_path,
        incidents,
        "duplicate-delivery",
        [call("money_check"), diagnose(RIGHT + " It is $12.34.", [("c1", "x")])],
    )
    assert by_name(right)["hypothesis_facts_in_results"]["passed"]


def test_a_wrong_cause_fails_the_rule(tmp_path, incidents):
    script = [
        call("money_check"),
        diagnose("Redis lost its idempotency keys, so the ledger posted events twice.", [("c1", "x")]),
    ]
    result = run_and_check(tmp_path, incidents, "duplicate-delivery", script)
    assert by_name(result)["names_the_cause"]["detail"] == "the rule says: misses the cause"


def test_missing_deploys_needs_it_to_say_what_it_could_not_check(tmp_path, incidents):
    script = [call("money_check"), call("recent_deploys"), diagnose(RIGHT, [("c1", "evt_000101 posted 2 times")])]
    result = run_and_check(tmp_path, incidents, "missing-deploys", script)
    assert not by_name(result)["says_what_it_could_not_check"]["passed"]


def test_missing_deploys_refuses_what_only_the_deploy_list_holds(tmp_path, incidents):
    script = [
        call("money_check"),
        call("recent_deploys"),
        diagnose(
            RIGHT + " A deploy set the naive handler; the deploy list could not be read, docker did not answer.",
            [("c1", "evt_000101 posted 2 times"), ("c2", "recent_deploys failed: docker did not answer")],
        ),
    ]
    result = run_and_check(tmp_path, incidents, "missing-deploys", script)
    checks = by_name(result)
    assert checks["says_what_it_could_not_check"]["passed"]
    assert not checks["claims_nothing_withheld"]["passed"] and "naive" in checks["claims_nothing_withheld"]["detail"]
    assert "naive" in checks["claims_nothing_withheld"]["withheld_facts"]


def test_a_run_without_a_diagnosis_fails(tmp_path, incidents):
    result = run_and_check(tmp_path, incidents, "duplicate-delivery", ["{", "{", "{"])
    assert not result["passed"] and result["stop_reason"] == "invalid_replies"


def test_facts_are_found_in_their_kinds():
    found = {
        f["said"]
        for f in check.facts(
            "evt_000101 and pay_000090 cost $1,234.50 at 2026-11-20T06:59:12, LEDGER_HANDLER=naive, provider-events/1@112"
        )
    }
    assert found == {
        "evt_000101",
        "pay_000090",
        "$1,234.50",
        "2026-11-20T06:59:12",
        "LEDGER_HANDLER=naive",
        "provider-events/1@112",
    }


def test_a_setting_at_the_end_of_a_sentence_keeps_its_value_only():
    # Found on the real data: "... with LEDGER_HANDLER=naive." once read as the value "naive." and failed the check.
    assert check.facts("deployed with LEDGER_HANDLER=naive.") == [{"kind": "setting", "said": "LEDGER_HANDLER=naive"}]
    assert check.facts("BAD_EVENTS: retry, and more")[0]["said"] == "BAD_EVENTS=retry"


def test_holds_reads_amounts_in_dollars_or_cents_and_settings_as_json():
    assert check.holds({"kind": "amount", "said": "$1,234.50"}, '"amount_cents": 123450')
    assert check.holds({"kind": "amount", "said": "$12.34"}, "($12.34)")
    assert not check.holds({"kind": "amount", "said": "$12.34"}, '"amount_cents": 123400')
    assert check.holds({"kind": "setting", "said": "LEDGER_HANDLER=naive"}, '{"LEDGER_HANDLER": "naive"}')
    assert not check.holds({"kind": "id", "said": "evt_00010"}, "evt_000101")


# Version 2 of the checks (written after the live runs; version 1 stays the default and is tested above).

MONEY = {
    "outcome": "failed",
    "invariants": [
        {
            "name": "posted_exactly_once",
            "passed": False,
            "evidence": [
                {
                    "event_id": "evt_1",
                    "type": "payment.succeeded",
                    "amount_cents": 1697,
                    "times_posted": 2,
                    "expected": 1,
                },
                {
                    "event_id": "evt_2",
                    "type": "payment.succeeded",
                    "amount_cents": 1929,
                    "times_posted": 2,
                    "expected": 1,
                },
                {
                    "event_id": "evt_3",
                    "type": "refund.succeeded",
                    "amount_cents": 875,
                    "times_posted": 3,
                    "expected": 1,
                },
            ],
        },
        {"name": "matches_provider", "passed": False, "evidence": [{"ledger_cents": 1, "provider_cents": 2}]},
    ],
}


def test_money_sums_are_the_extra_money_of_some_findings():
    sums = check.money_sums(json.dumps(MONEY))
    assert 1697 + 1929 in sums  # the two payments counted twice
    assert 1750 in sums  # the refund's two extra postings
    assert 1697 + 1929 - 1750 in sums  # all of them, signed: what the books are off by
    assert 1697 + 1929 + 1750 in sums  # all of them, by size
    assert 9999 not in sums and 1697 * 3 not in sums
    assert len(sums) <= 2 * (2**3 - 1)
    assert check.money_sums("not json") == set() and check.money_sums('{"outcome": "could not run"}') == set()


def test_version_2_takes_a_sum_of_findings_and_still_refuses_any_other_amount(tmp_path, incidents):
    folder = incidents / "duplicate-delivery"
    money = json.loads((folder / "money_check.json").read_text())
    money["invariants"][0]["evidence"] = MONEY["invariants"][0]["evidence"]
    (folder / "money_check.json").write_text(json.dumps(money))
    hypothesis = RIGHT + " The two payments counted twice add $36.26."
    script = [call("money_check"), diagnose(hypothesis, [("c1", "two payments counted twice add $36.26")])]
    run_and_check(tmp_path, incidents, "duplicate-delivery", script)
    run = tmp_path / "duplicate-delivery"
    one, two = check.check_run(run, incidents, 1), check.check_run(run, incidents, 2)
    assert not by_name(one)["hypothesis_facts_in_results"]["passed"]  # version 1: $36.26 is in no result
    assert (
        by_name(two)["hypothesis_facts_in_results"]["passed"] and by_name(two)["cited_results_hold_the_facts"]["passed"]
    )
    assert two["version"] == 2 and "version" not in one
    (run / "trace.jsonl").write_text((run / "trace.jsonl").read_text().replace("$36.26", "$36.27"))
    assert not by_name(check.check_run(run, incidents, 2))["hypothesis_facts_in_results"]["passed"]


def test_version_2_reads_the_fix_as_a_proposal_not_a_claim(tmp_path, incidents):
    script = [
        call("money_check"),
        diagnose(RIGHT, [("c1", "evt_000101 posted 2 times")], fix_summary="Restore LEDGER_HANDLER=idempotent."),
    ]
    run_and_check(tmp_path, incidents, "duplicate-delivery", script)
    run = tmp_path / "duplicate-delivery"
    assert not by_name(check.check_run(run, incidents, 1))["hypothesis_facts_in_results"]["passed"]
    assert by_name(check.check_run(run, incidents, 2))["hypothesis_facts_in_results"]["passed"]


def test_version_2_still_refuses_a_fact_cited_to_the_wrong_call(tmp_path, incidents):
    script = [
        call("money_check"),
        call("search_logs", {"contains": "evt_000101"}),
        diagnose(RIGHT, [("c1", "evt_000101 posted 2 times"), ("c2", "pay_000090 was charged twice")]),
    ]
    run_and_check(tmp_path, incidents, "duplicate-delivery", script)
    assert not by_name(check.check_run(tmp_path / "duplicate-delivery", incidents, 2))["cited_results_hold_the_facts"][
        "passed"
    ]


def test_no_third_version():
    with pytest.raises(ValueError):
        check.check_run(Path("."), version=3)


# The read_only check, and a run whose two records disagree. These are written by hand into a run's files: the
# loop itself can neither run a tool that writes nor apply a fix.


def rewrite(path, change):
    records = [json.loads(line) for line in path.read_text().splitlines()]
    for record in records:
        change(record)
    path.write_text("".join(json.dumps(record) + "\n" for record in records))


def renamed(old, new):
    def change(record):
        for place in (record.get("ran"), record.get("proposed"), record):
            if isinstance(place, dict) and place.get("tool") == old and (place is not record or "seq" in record):
                place["tool"] = new

    return change


def a_scripted_run(tmp_path, incidents):
    script = [call("money_check"), diagnose(RIGHT, [("c1", "evt_000101 posted 2 times")])]
    assert run_and_check(tmp_path, incidents, "duplicate-delivery", script)["passed"]
    return tmp_path / "duplicate-delivery"


def test_the_read_only_check_fails_when_a_tool_that_writes_ran(tmp_path, incidents):
    run = a_scripted_run(tmp_path, incidents)
    rewrite(run / loop.TRACE, renamed("money_check", "rollback_deploy"))
    rewrite(run / loop.TOOL_RESULTS, renamed("money_check", "rollback_deploy"))
    for version in (1, 2):
        read_only = by_name(check.check_run(run, incidents, version))["read_only"]
        assert not read_only["passed"] and "other tools: ['rollback_deploy']" in read_only["detail"]


def test_the_read_only_check_fails_when_the_fix_was_applied(tmp_path, incidents):
    run = a_scripted_run(tmp_path, incidents)

    def applied(record):
        if record.get("kind") == "stop":
            record["diagnosis"]["proposed_fix"]["applied"] = True

    rewrite(run / loop.TRACE, applied)
    read_only = by_name(check.check_run(run, incidents))["read_only"]
    assert not read_only["passed"] and "applied: True" in read_only["detail"]


def test_a_run_whose_trace_and_recording_disagree_is_refused(tmp_path, incidents):
    run = a_scripted_run(tmp_path, incidents)
    rewrite(run / loop.TOOL_RESULTS, renamed("money_check", "search_logs"))
    with pytest.raises(ValueError, match="call c1 differs between the trace and the recording"):
        check.check_run(run, incidents)


def test_a_hidden_character_in_a_tool_argument_does_not_stop_the_check(tmp_path, incidents):
    # The trace keeps arguments cleaned, the recording as they came: once read as a disagreement, and a crash.
    script = [
        call("search_logs", {"contains": "evt­_000101"}),
        diagnose(RIGHT, [("c1", "it shows the redeliveries")]),
    ]
    result = run_and_check(tmp_path, incidents, "duplicate-delivery", script)
    assert result["tool_calls"] == 1 and by_name(result)["citations_ran"]["passed"]


# The fixes version 2 took on 2026-11-21 UTC, after a code review. Version 1 is kept as committed: its gaps are tested
# here too, so that it cannot change unnoticed.


def test_version_2_reads_times_with_no_date_and_version_1_reads_none():
    text = "restarted at 07:41:49.752Z, then at 07:42, and once on 2026-11-20T07:43:10Z"
    assert [f["said"] for f in check.facts(text) if f["kind"] == "time"] == ["2026-11-20T07:43:10"]
    assert {f["said"] for f in check.facts(text, 2) if f["kind"] == "time"} == {
        "07:41:49.752",
        "07:42",
        "2026-11-20T07:43:10",
    }


def test_version_2_reads_offsets_in_their_short_forms_and_with_a_full_width_at_sign():
    def offsets(text, version=2):
        return sorted(f["said"] for f in check.facts(text, version) if f["kind"] == "offset")

    assert offsets("provider-events/2＠95") == ["provider-events/2@95"]
    assert offsets("provider-events/2＠95", version=1) == []  # version 1 skipped it
    assert offsets("offsets 2@95-97") == ["provider-events/2@95", "provider-events/2@96", "provider-events/2@97"]
    assert offsets("the message 0@229, after provider-events/0@228") == [
        "provider-events/0@228",
        "provider-events/0@229",
    ]
    assert offsets("offsets 95, 96 and 97") == []  # written in words: not read, a known limit


def test_version_2_finds_cents_only_as_the_value_of_a_cents_key():
    result = '{"at": "2026-11-20T07:42:10Z", "matched": 351, "difference_cents": -1750, "amount_cents": "1697"}'
    for made_up in ("$20.26", "$3.51"):  # the year, and a count
        assert check.holds({"kind": "amount", "said": made_up}, result)  # version 1 took digits anywhere
        assert not check.holds({"kind": "amount", "said": made_up}, result, 2)
    assert check.holds({"kind": "amount", "said": "$17.50"}, result, 2)
    assert check.holds({"kind": "amount", "said": "$16.97"}, result, 2)
    assert check.holds({"kind": "amount", "said": "$116.97"}, "off by +$116.97", 2)


def test_version_2_holds_times_offsets_and_settings_as_a_log_line_gives_them():
    fields = {"event_id": "evt_000308", "offset": "97", "partition": "2", "bad_events": "dead_letter"}
    line = json.dumps({"time": "2026-11-20T07:41:43.478Z", "service": "webhook-receiver", "fields": fields})
    assert check.holds({"kind": "time", "said": "07:41:43.478"}, line, 2)
    assert check.holds({"kind": "time", "said": "07:41"}, line, 2)
    assert not check.holds({"kind": "time", "said": "07:41:44"}, line, 2)
    assert check.holds({"kind": "offset", "said": "provider-events/2@97"}, line, 2)
    assert not check.holds({"kind": "offset", "said": "provider-events/2@96"}, line, 2)
    assert check.holds({"kind": "setting", "said": "BAD_EVENTS=dead_letter"}, line, 2)
    assert not check.holds({"kind": "setting", "said": "BAD_EVENTS=dead_letter"}, line)  # version 1: case matters


def test_version_2_reads_the_whole_alert_so_its_time_may_be_restated(tmp_path, incidents):
    hypothesis = RIGHT + " The alert fired at 2026-11-20T07:00:30Z."
    script = [call("money_check"), diagnose(hypothesis, [("c1", "evt_000101 posted 2 times")])]
    run_and_check(tmp_path, incidents, "duplicate-delivery", script)
    run = tmp_path / "duplicate-delivery"
    assert not by_name(check.check_run(run, incidents, 1))["hypothesis_facts_in_results"]["passed"]
    assert by_name(check.check_run(run, incidents, 2))["hypothesis_facts_in_results"]["passed"]


@pytest.mark.parametrize("said", ["06:58:30", "2026-11-20 06:58"])
def test_version_2_catches_the_withheld_deploy_time_in_more_forms(tmp_path, incidents, said):
    hypothesis = RIGHT + f" The ledger consumer was redeployed at {said}, though the deploy list did not answer."
    script = [call("money_check"), call("recent_deploys"), diagnose(hypothesis, [("c1", "evt_000101 posted 2 times")])]
    run_and_check(tmp_path, incidents, "missing-deploys", script)
    run = tmp_path / "missing-deploys"
    assert by_name(check.check_run(run, incidents, 1))["claims_nothing_withheld"]["passed"]  # version 1 missed it
    withheld = by_name(check.check_run(run, incidents, 2))["claims_nothing_withheld"]
    assert not withheld["passed"] and "06:58:30" in withheld["withheld_facts"]


def test_money_sums_survive_shapes_a_money_check_never_gives():
    for odd in ('{"invariants": null}', '{"invariants": [7]}', '{"invariants": [{"evidence": ["x", 3, null]}]}', "[1]"):
        assert check.money_sums(odd) == set()
