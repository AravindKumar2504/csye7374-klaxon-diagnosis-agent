"""The scripted mock model: replies worked out by code, so the agent runs offline with no model at all.

New for this mini-project. It stands in for the model in the offline demo and in the tests. Its replies are not a
model's: each is worked out by a few lines of Python from what the tools returned so far, always in the same order
(the money check, then the logs, then the deploys). So a run on it shows the loop, the tools, the trace and the checks
working, offline and the same every time. It shows nothing of how a model chooses or diagnoses: that is what the live
runs are for. The trace of every such run says so (its source is "scripted": no model was called).
"""

from __future__ import annotations

import json
from collections.abc import Sequence

from diagnosis_agent.cases import Case
from diagnosis_agent.loop import UNTRUSTED
from diagnosis_agent.model import Message, ScriptedModel


def results(turns: Sequence[Message]) -> dict[str, dict]:
    """The tool results so far, by call id, read back from the user turns the loop wrote (diagnosis_agent.loop's
    observation(): a line saying which call, the untrusted-data note, then the result as one JSON line)."""
    found: dict[str, dict] = {}
    for turn in turns:
        if turn.role != "user":
            continue
        lines = turn.content.split("\n")
        for i, line in enumerate(lines[:-1]):
            if line == UNTRUSTED:
                document = json.loads(lines[i + 1])
                found[document["call_id"]] = document
    return found


def call(tool: str, arguments: dict, why: str) -> str:
    return json.dumps({"action": "call_tool", "why": why, "tool": tool, "arguments": arguments})


def diagnose(why: str, hypothesis: str, evidence: list[tuple[str, str]], confidence: float, fix: dict) -> str:
    return json.dumps(
        {
            "action": "diagnose",
            "why": why,
            "diagnosis": {
                "hypothesis": hypothesis,
                "evidence": [{"call_id": call_id, "shows": shows} for call_id, shows in evidence],
                "confidence": confidence,
                "proposed_fix": fix,
                "blast_radius": "The ledger consumer only: while it restarts, events wait on Kafka and are booked when it is back.",
                "rollback_plan": "Deploy the ledger consumer again with the settings it runs now.",
            },
        }
    )


def _ledger_deploy(document: dict) -> dict | None:
    if document["is_error"]:
        return None
    return next((d for d in document["result"].get("deploys", []) if d.get("service") == "ledger-consumer"), None)


def counted_twice(system: str, turns: Sequence[Message]) -> str:
    """Incident 1, and the missing-evidence case: the money check, the logs of the event it names, the deploys."""
    found = results(turns)
    if "c1" not in found:
        return call("money_check", {}, "Confirm the alert's symptom first: run the money check.")
    money = found["c1"]["result"]
    twice = next(
        (inv for inv in money.get("invariants", []) if inv.get("name") == "posted_exactly_once" and not inv["passed"]),
        None,
    )
    event = twice["evidence"][0] if twice and twice.get("evidence") else {}
    event_id = event.get("event_id", "")
    if "c2" not in found:
        return call(
            "search_logs",
            {"contains": event_id, "minutes": 30, "limit": 10},
            f"See what each service did with {event_id}.",
        )
    if "c3" not in found:
        return call("recent_deploys", {}, "See what changed: a deploy can remove the ledger's dedupe.")
    lines = found["c2"]["result"].get("lines", [])
    posted = sum(1 for line in lines if line["service"] == "ledger-consumer" and line["message"] == "posted")
    queued = sum(1 for line in lines if line["service"] == "webhook-receiver")
    ledger = _ledger_deploy(found["c3"])
    evidence = [
        ("c1", f"The money check failed: {event_id} posted {event.get('times_posted')} times, expected 1."),
        (
            "c2",
            f"The webhook receiver queued {event_id} {queued} times, and the ledger consumer posted it {posted} times.",
        ),
    ]
    if ledger is None:
        deploy = (
            " The deploy list could not be read (docker did not answer), so which deploy brought this could not be"
            " checked."
        )
        evidence.append(("c3", "recent_deploys failed: docker did not answer, so the deploy list could not be read."))
        confidence = 0.6
    else:
        handler = ledger.get("settings", {}).get("LEDGER_HANDLER")
        deploy = f" recent_deploys shows the ledger consumer deployed at {ledger.get('deployed_at')} with LEDGER_HANDLER={handler}."
        evidence.append(
            ("c3", f"The ledger consumer was deployed at {ledger.get('deployed_at')} with LEDGER_HANDLER={handler}.")
        )
        confidence = 0.8
    hypothesis = (
        f"The ledger consumer books some provider events twice: {event_id} was posted {event.get('times_posted')}"
        " times. The provider redelivered its webhook, as it does when an acknowledgement is lost, and the ledger"
        f" consumer posted each delivery instead of skipping it as already processed: its dedupe is gone.{deploy}"
    )
    fix = {
        "kind": "rollback",
        "target": "ledger-consumer",
        "summary": "Roll the ledger consumer back to its idempotent handler, then reverse each double posting.",
    }
    return diagnose("The money check, the logs and the deploys agree.", hypothesis, evidence, confidence, fix)


def stuck(system: str, turns: Sequence[Message]) -> str:
    """Incident 2: the money check, the ledger consumer's errors, the deploys."""
    found = results(turns)
    if "c1" not in found:
        return call("money_check", {}, "Confirm the alert's symptom first: run the money check.")
    if "c2" not in found:
        return call(
            "search_logs",
            {"service": "ledger-consumer", "level": "ERROR", "limit": 5},
            "The books are behind: see what the ledger consumer is failing on.",
        )
    if "c3" not in found:
        return call("recent_deploys", {}, "See what changed in the ledger consumer.")
    money = found["c1"]["result"]
    lines = found["c2"]["result"].get("lines", [])
    newest = lines[0] if lines else {"fields": {}}
    fields = newest.get("fields", {})
    ledger = _ledger_deploy(found["c3"]) or {}
    bad = ledger.get("settings", {}).get("BAD_EVENTS")
    hypothesis = (
        f"The ledger consumer is stuck on one message it cannot book ({fields.get('source')}:"
        f" {fields.get('error')}), and it retries that same message forever (attempt {fields.get('attempt')} so"
        " far), so it books nothing behind it. A deploy of the ledger consumer set"
        f" BAD_EVENTS={bad}, which retries an unreadable message instead of dead-lettering it."
    )
    evidence = [
        ("c1", f"The money check could not run: {money.get('reason')}."),
        (
            "c2",
            f"The ledger consumer logs '{newest.get('message')}' for {fields.get('source')}, attempt {fields.get('attempt')}.",
        ),
        ("c3", f"The ledger consumer was deployed at {ledger.get('deployed_at')} with BAD_EVENTS={bad}."),
    ]
    fix = {
        "kind": "rollback",
        "target": "ledger-consumer",
        "summary": "Roll the ledger consumer back to dead-lettering, so the poison message goes to the dead-letter topic.",
    }
    return diagnose("The money check, the errors and the deploys agree.", hypothesis, evidence, 0.8, fix)


POLICIES = {"duplicate-delivery": counted_twice, "missing-deploys": counted_twice, "poison-message": stuck}


def model_for(case: Case, steps: int = 10) -> ScriptedModel:
    """The scripted model for a case: the same policy at every step, which reads the results so far."""
    return ScriptedModel([POLICIES[case.name]] * steps, model="scripted")
