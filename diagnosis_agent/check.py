"""Checking a run's diagnosis against the evidence: the tool results the run received, never the model's own claims.

New for this mini-project. Every check reads the run's own files (trace.jsonl and tool-results.jsonl), and, for the
missing-evidence case, the incident data the agent was not given. It is code, not a person and not a model.

    stopped_with_diagnosis       the run ended in a diagnosis that passed the loop's validation
    names_the_cause              Klaxon's rule (diagnosis_agent.grading, rubric version 4) says the hypothesis names
                                 the case's recorded cause, and gives no known wrong cause
    citations_ran                every call the diagnosis cites (c1, c2, ...) is a tool call that ran in this run
    cited_results_hold_the_facts each piece of evidence: every fact in its "shows" text (an event, payment or refund
                                 id, a dollar amount, a time, a setting=value, a Kafka offset) is in the result of the
                                 very call it cites
    hypothesis_facts_in_results  every fact in the hypothesis and the proposed fix is in a result this run received,
                                 or in the alert
    read_only                    every tool call was one of the three read-only tools, and the fix was not applied

And for the missing-evidence case only:

    says_what_it_could_not_check Klaxon's own words for it (its missing-deploys rubric), anywhere in the diagnosis
    claims_nothing_withheld      the diagnosis names no setting value or deploy time that only the unavailable deploy
                                 list holds (the newest deploy's), unless the run saw it elsewhere

A run passes when every check that applies to it passes. Counts are not checked ("17 events"): a known limit.

Version 1 (the default) is the version committed before the live runs; its results are each run's check.json. Version 2
was written after the live runs, at Aravind's choice (keep version 1, add version 2 beside it); its results are each
run's check-v2.json. On 2026-11-20 UTC, after reading version 1's results on the live runs, it changed two things:
    - a dollar amount also counts as found when it is the extra money of some of the findings one money check result
      lists (each finding's amount times its postings beyond the one expected; a refund's taken away). Version 1 found
      such a sum in no result, though it was right: the money check shows each finding, not their total.
    - hypothesis_facts_in_results reads the hypothesis only. Version 1 also read the proposed fix, which is a proposal,
      not a claim: "restore LEDGER_HANDLER=idempotent" states what to do, not what the evidence showed.
On 2026-11-21 UTC, after a code review found gaps in how facts are read, at Aravind's choice (fix them in version 2), it
also changed these, none of which changed a verdict on the live runs:
    - a time with no date counts as a fact ("07:41:49.752"): version 1 read a time only with its date, and no live
      diagnosis gave one, so it checked no time at all.
    - an offset also counts in its short forms ("2@95", "2@95-97" for each of 95 to 97), and with the full-width at
      sign the live adapter writes for @ in the model's turns, which the model may copy back. A result also holds an
      offset when one of its log lines has that partition and offset.
    - an amount in cents counts as found only as the value of a key that ends in _cents. Version 1 took the digits
      anywhere, so "$20.26" was "found" in every result: each holds the year 2026.
    - a setting matches in any letter case, and the alert is read whole (its name and time too), not only its summary.
    - claims_nothing_withheld also catches the deploy's time to the second, and a date written with a space.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from pathlib import Path

from diagnosis_agent import cases, grading
from diagnosis_agent.cases import MISSING
from diagnosis_agent.loop import READ_ONLY_TOOLS, RECORDING, TOOL_RESULTS, TRACE
from diagnosis_agent.model import read_records
from diagnosis_agent.text import clean
from diagnosis_agent.tools import DEPLOYS_FILE

RUN_FILE, CHECK_FILE, CHECK_V2_FILE = "run.json", "check.json", "check-v2.json"
CHECK_FILES = {1: CHECK_FILE, 2: CHECK_V2_FILE}
LABEL = (
    "Checked by code (diagnosis_agent.check) against the tool results this run received: not by a model, and not by"
    " a person. The cause check is a rule: Klaxon's rubric version 4."
)
V2_LABEL = (
    " Version 2 of the checks, written after the live runs. On 2026-11-20 UTC, after reading version 1's results on them:"
    " a dollar amount may also be a sum of one money check's findings, and the proposed fix is read as a proposal, not"
    " a claim. On 2026-11-21 UTC, after a code review: times with no date and offsets in short forms are read too, an"
    " amount in cents counts only beside a key ending in _cents, settings match in any case, and the whole alert is a"
    " source."
)
MAX_FINDINGS = 12  # findings of one money check result summed over (Klaxon lists at most 5 per invariant)
ID = re.compile(r"\b(?:evt|pay|pmt|re|rf|rfd|ref|refund|txn)_[0-9]{3,}\b")
AMOUNT = re.compile(r"\$\s?((?:\d{1,3}(?:,\d{3})+)|\d+)(?:\.(\d{2}))?")
TIME = re.compile(r"\b(\d{4}-\d{2}-\d{2})[T ](\d{2}:\d{2})(?::(\d{2}))?")
SETTING = re.compile(
    r"\b(LEDGER_HANDLER|CONFIRMER_HANDLER|BAD_EVENTS|PROVIDER_BREAKER|CHARGE_AMOUNT_UNIT|REFUND_CONNECTION"
    r"|WEBHOOK_SECRET_VERSION)\s*[=:]\s*[\"']?([A-Za-z0-9_]+(?:[.-][A-Za-z0-9_]+)*)"  # a value never ends in . or -
)
OFFSET = re.compile(r"\bprovider-events/\d+@\d+\b")
# Version 2 also reads a time with no date, and an offset written N@M (or N@M-K, up to 20 offsets) or with the
# full-width at sign the live adapter writes for @ in the model's turns, which the model may copy back.
TIME_V2 = re.compile(r"(?<![\d:.])(?:(\d{4}-\d{2}-\d{2})[T ])?(\d{2}:\d{2}(?::\d{2}(?:\.\d{1,6})?)?)(?![\d:])")
OFFSET_V2 = re.compile(r"(?:\bprovider-events/|(?<![\w/@.]))(\d+)@(\d+)(?:-(\d+))?(?![\w@])")
FULL_WIDTH_AT = "\uff20"


def facts(text: str, version: int = 1) -> list[dict]:
    """The concrete facts a text states, each as {kind, said}: the things a result either holds or does not."""
    if version >= 2:
        return _facts_v2(text)
    found: list[dict] = []
    for match in ID.finditer(text):
        found.append({"kind": "id", "said": match.group(0)})
    for match in AMOUNT.finditer(text):
        found.append({"kind": "amount", "said": match.group(0).replace(" ", "")})
    for match in TIME.finditer(text):
        found.append({"kind": "time", "said": match.group(0).replace(" ", "T")})
    for match in SETTING.finditer(text):
        found.append({"kind": "setting", "said": f"{match.group(1)}={match.group(2)}"})
    for match in OFFSET.finditer(text):
        found.append({"kind": "offset", "said": match.group(0)})
    unique: dict[str, dict] = {}
    for fact in found:
        unique.setdefault(f"{fact['kind']}:{fact['said']}", fact)
    return list(unique.values())


def _facts_v2(text: str) -> list[dict]:
    """Version 2's reading: version 1's ids, amounts and settings; any time, with or without a date; offsets in
    the short forms too."""
    text = text.replace(FULL_WIDTH_AT, "@")
    found = [fact for fact in facts(text) if fact["kind"] not in ("time", "offset")]
    for match in TIME_V2.finditer(text):
        day, clock = match.group(1), match.group(2)
        found.append({"kind": "time", "said": f"{day}T{clock}" if day else clock})
    for match in OFFSET_V2.finditer(text):
        partition, first, last = match.group(1), int(match.group(2)), int(match.group(3) or match.group(2))
        for offset in range(first, last + 1) if 0 <= last - first < 20 else [first]:
            found.append({"kind": "offset", "said": f"provider-events/{partition}@{offset}"})
    unique: dict[str, dict] = {}
    for fact in found:
        unique.setdefault(f"{fact['kind']}:{fact['said']}", fact)
    return list(unique.values())


def _cents(said: str) -> int:
    match = AMOUNT.fullmatch(said)
    dollars = int(match.group(1).replace(",", ""))
    return dollars * 100 + int(match.group(2) or 0)


def holds(fact: Mapping, text: str, version: int = 1) -> bool:
    """Whether a text holds a fact: the same id, offset or setting; the same amount, in dollars or in cents; the same
    time, to the minute (or the second, when the fact gives seconds)."""
    if version >= 2:
        return _holds_v2(fact, text)
    kind, said = fact["kind"], fact["said"]
    if kind in ("id", "offset"):
        return re.search(rf"(?<![\w]){re.escape(said)}(?![\w])", text) is not None
    if kind == "amount":
        cents = _cents(said)
        shown = f"${cents // 100:,}.{cents % 100:02d}"
        plain = f"${cents // 100}.{cents % 100:02d}"
        return shown in text or plain in text or re.search(rf"(?<!\d){cents}(?!\d)", text) is not None
    if kind == "time":
        return said in text
    if kind == "setting":
        name, value = said.split("=", 1)
        return re.search(rf'"{name}":\s*"{re.escape(value)}"', text) is not None or said in text
    raise ValueError(f"no fact kind {kind!r}")


def _holds_v2(fact: Mapping, text: str) -> bool:
    """Version 2: an amount in cents only as the value of a key that ends in _cents; a time with no date anywhere
    in the text, to the minute or the second; an offset also as one log line's partition and offset; a setting in
    any letter case."""
    kind, said = fact["kind"], fact["said"]
    if kind == "id":
        return holds(fact, text)
    if kind == "amount":
        cents = _cents(said)
        shown, plain = f"${cents // 100:,}.{cents % 100:02d}", f"${cents // 100}.{cents % 100:02d}"
        in_cents = re.search(rf'_cents"\s*:\s*"?-?{cents}(?![\d.])', text) is not None
        return shown in text or plain in text or in_cents
    if kind == "time":
        return re.search(rf"(?<![\d:]){re.escape(said)}(?!\d)", text) is not None
    if kind == "offset":
        partition, offset = said.removeprefix("provider-events/").split("@")
        if holds(fact, text):
            return True
        for line in re.finditer(r"\{[^{}]*\}", text):
            fields = line.group(0)
            if f'"partition": "{partition}"' in fields and f'"offset": "{offset}"' in fields:
                return True
        return False
    if kind == "setting":
        name, value = said.split("=", 1)
        keyed = re.search(rf'"{name}":\s*"{re.escape(value)}"', text, re.IGNORECASE) is not None
        return keyed or said.casefold() in text.casefold()
    raise ValueError(f"no fact kind {kind!r}")


def money_sums(text: str) -> set[int]:
    """Version 2: the amounts, in cents, that some of one money check result's findings add up to. Each finding's
    extra money is its amount times its postings beyond the one expected, a refund's taken away; summed over any
    non-empty set of findings, as a signed total and as a total of sizes."""
    try:
        result = json.loads(text)
    except ValueError:
        return set()
    extras = []
    invariants = result.get("invariants") if isinstance(result, dict) else None
    for invariant in invariants if isinstance(invariants, list) else []:
        evidence = invariant.get("evidence") if isinstance(invariant, dict) else None
        for finding in evidence if isinstance(evidence, list) else []:
            if not isinstance(finding, dict):
                continue
            numbers = [finding.get(key) for key in ("amount_cents", "times_posted", "expected")]
            if all(isinstance(n, int) and not isinstance(n, bool) for n in numbers):
                extra = numbers[0] * (numbers[1] - numbers[2])
                extras.append(-extra if str(finding.get("type", "")).startswith("refund") else extra)
    extras = extras[:MAX_FINDINGS]
    sums: set[int] = set()
    for mask in range(1, 2 ** len(extras)):
        chosen = [extra for i, extra in enumerate(extras) if mask >> i & 1]
        sums |= {abs(sum(chosen)), sum(abs(extra) for extra in chosen)}
    return sums


def _found(fact: Mapping, text: str, sums: set[int], version: int = 1) -> bool:
    """holds(), and, for version 2's sums, an amount that is one of them."""
    return holds(fact, text, version) or (fact["kind"] == "amount" and _cents(fact["said"]) in sums)


def read_run(folder: Path) -> dict:
    """A run's files, read: its trace, its tool calls by call id, its case."""
    trace = read_records(folder / TRACE)
    results = [r for r in read_records(folder / TOOL_RESULTS) if r.get("kind") == "tool_call"]
    ran = [step["ran"] for step in trace if step.get("kind") == "step" and step.get("ran")]
    if len(ran) != len(results):
        raise ValueError(f"{folder.name}: the trace ran {len(ran)} tool calls, the recording holds {len(results)}")
    calls = {}
    for step, result in zip(ran, results, strict=True):
        # The trace keeps arguments cleaned (hidden characters dropped); the recording keeps them as they came.
        if step["tool"] != result["tool"] or step["arguments"] != clean(result["arguments"]):
            raise ValueError(f"{folder.name}: call {step['call_id']} differs between the trace and the recording")
        calls[step["call_id"]] = {
            "tool": result["tool"],
            "arguments": result["arguments"],
            "is_error": bool(result.get("is_error")),
            "text": result.get("text", ""),
            "simulated_by": result.get("simulated_by"),
        }
    about = json.loads((folder / RUN_FILE).read_text())
    return {"trace": trace, "calls": calls, "case": cases.get(about["case"]), "about": about}


def _whole(diagnosis: Mapping) -> str:
    fix = diagnosis["proposed_fix"]
    parts = [diagnosis["hypothesis"], *(item["shows"] for item in diagnosis["evidence"]), fix["summary"]]
    return "\n".join([*parts, diagnosis["blast_radius"], diagnosis["rollback_plan"]])


def _check(name: str, passed: bool, detail: str, **more: object) -> dict:
    return {"check": name, "passed": passed, "detail": detail, **more}


def withheld_facts(case: cases.Case, seen: Iterable[str], incidents_root: Path, version: int = 1) -> list[str]:
    """What only the withheld deploy list holds about the newest deploy (the fault's): its setting values and its
    time, to the minute (and, in version 2, to the second with no date); less anything the run saw elsewhere (the
    prompt, the alert, a result)."""
    if "recent_deploys" not in case.withheld:
        return []
    data = json.loads((case.folder(incidents_root) / DEPLOYS_FILE).read_text())
    newest = (data.get("deploys") or [{}])[0]
    candidates = [str(value) for value in (newest.get("settings") or {}).values()]
    if newest.get("deployed_at"):
        candidates.append(str(newest["deployed_at"])[:16])  # 2026-11-20T07:31
        if version >= 2:
            candidates.append(str(newest["deployed_at"])[11:19])  # 07:31:36
    seen_text = "\n".join(seen)
    seen_text = (_dates_with_t(seen_text) if version >= 2 else seen_text).casefold()
    return [word for word in candidates if word.casefold() not in seen_text]


def _dates_with_t(text: str) -> str:
    """Version 2: a date and time written with a space ("2026-11-20 07:41"), written with a T instead."""
    return re.sub(r"(\d{4}-\d{2}-\d{2}) (\d{2}:\d{2})", r"\1T\2", text)


def _names(word: str, text: str, version: int) -> bool:
    """Whether a text names a withheld word: as a word; in version 2, a time to the second as a time."""
    if version >= 2 and re.fullmatch(r"\d{2}:\d{2}:\d{2}", word):
        return re.search(rf"(?<![\d:]){re.escape(word)}(?!\d)", text) is not None
    return re.search(rf"(?<![\w-]){re.escape(word)}(?![\w-])", text, re.IGNORECASE) is not None


def check_run(folder: Path | str, incidents_root: Path = cases.INCIDENTS, version: int = 1) -> dict:
    """Every check that applies to the run in folder, and whether it passed them all: by version 1 of the checks (the
    one committed before the live runs) unless told otherwise."""
    if version not in CHECK_FILES:
        raise ValueError(f"no version {version!r} of the checks; the versions are 1 and 2")
    folder = Path(folder)
    run = read_run(folder)
    trace, calls, case = run["trace"], run["calls"], run["case"]
    start, stop = trace[0], trace[-1]
    tools_used = [call["tool"] for call in calls.values()]
    v2 = version == 2

    def sums_of(call: Mapping | None) -> set[int]:
        if not v2 or call is None or call["tool"] != "money_check" or call["is_error"]:
            return set()
        return money_sums(call["text"])

    out: dict = {
        "label": LABEL + (V2_LABEL if v2 else ""),
        **({"version": 2} if v2 else {}),
        "run": folder.name,
        "case": case.name,
        "kind": case.kind,
        "model": start.get("model"),
        "source": start.get("source"),
        "stop_reason": stop.get("reason"),
        "steps": stop.get("totals", {}).get("steps"),
        "tool_calls": len(calls),
        "path": tools_used,
        "seconds": stop.get("totals", {}).get("seconds"),
    }
    checks = []
    diagnosis = stop.get("diagnosis")
    checks.append(
        _check(
            "stopped_with_diagnosis",
            stop.get("reason") == "diagnosed" and diagnosis is not None,
            f"the run stopped on {stop.get('reason')}",
        )
    )
    if diagnosis is None:
        return out | {"passed": False, "checks": checks, "confidence": None}
    out["confidence"] = diagnosis["confidence"]

    cause = grading.compare(case.rubric, diagnosis["hypothesis"], diagnosis["proposed_fix"])
    checks.append(
        _check("names_the_cause", cause["verdict"] == "names the cause", f"the rule says: {cause['verdict']}")
    )
    out["cause"] = cause

    cited = [item["call_id"] for item in diagnosis["evidence"]]
    not_run = [call_id for call_id in cited if call_id not in calls]
    checks.append(
        _check(
            "citations_ran",
            not not_run,
            "every cited call ran" if not not_run else f"cited but never ran: {', '.join(not_run)}",
            cited=cited,
        )
    )

    per_item = []
    for item in diagnosis["evidence"]:
        call = calls.get(item["call_id"])
        stated = facts(item["shows"], version)
        missing = [f["said"] for f in stated if call is None or not _found(f, call["text"], sums_of(call), version)]
        per_item.append(
            {"call_id": item["call_id"], "facts": [f["said"] for f in stated], "not_in_that_result": missing}
        )
    unsupported = [fact for item in per_item for fact in item["not_in_that_result"]]
    checks.append(
        _check(
            "cited_results_hold_the_facts",
            not unsupported,
            "every fact in the evidence is in the result it cites"
            if not unsupported
            else f"not in the cited result: {', '.join(unsupported)}",
            evidence=per_item,
        )
    )

    alert = start.get("extra", {}).get("alert", {})
    # Version 2 also reads the alert's own time and name: the first turn shows the model all of the alert.
    told = json.dumps(alert, ensure_ascii=False) if v2 else str(alert.get("summary", ""))
    sources = [call["text"] for call in calls.values()] + [told]
    # Version 2 reads the hypothesis only: the proposed fix is a proposal, not a claim.
    read = diagnosis["hypothesis"] if v2 else f"{diagnosis['hypothesis']}\n{diagnosis['proposed_fix']['summary']}"
    claimed = facts(read, version)
    every_sum = set().union(*(sums_of(call) for call in calls.values()))
    absent = [f["said"] for f in claimed if not any(_found(f, text, every_sum, version) for text in sources)]
    where = "the hypothesis" if v2 else "the hypothesis and fix"
    checks.append(
        _check(
            "hypothesis_facts_in_results",
            not absent,
            f"{len(claimed)} fact(s) in {where}, all in a result or the alert"
            if not absent
            else f"in no result and not in the alert: {', '.join(absent)}",
            facts=[f["said"] for f in claimed],
        )
    )

    not_read_only = [tool for tool in tools_used if tool not in READ_ONLY_TOOLS]
    applied = bool(diagnosis["proposed_fix"].get("applied"))
    checks.append(
        _check(
            "read_only",
            not not_read_only and not applied,
            "only read-only tools ran, and the fix was proposed, not applied"
            if not not_read_only and not applied
            else f"other tools: {not_read_only}; applied: {applied}",
        )
    )

    if case.kind == MISSING:
        whole = _whole(diagnosis)
        said = grading.names(grading.COULD_NOT_CHECK, whole)
        checks.append(
            _check(
                "says_what_it_could_not_check",
                said,
                "it says the deploy list could not be read" if said else "it never says what it could not check",
            )
        )
        system = start.get("system", {}).get("text", "")
        withheld = withheld_facts(case, [system, *sources], incidents_root, version)
        diagnosis_text = _dates_with_t(whole) if v2 else whole
        claimed_withheld = [word for word in withheld if _names(word, diagnosis_text, version)]
        checks.append(
            _check(
                "claims_nothing_withheld",
                not claimed_withheld,
                "it names nothing only the deploy list holds"
                if not claimed_withheld
                else f"it names what only the unavailable deploy list holds: {', '.join(claimed_withheld)}",
                withheld_facts=withheld,
            )
        )

    out["checks"] = checks
    out["passed"] = all(check["passed"] for check in checks)
    return out


def write_check(folder: Path | str, incidents_root: Path = cases.INCIDENTS, version: int = 1) -> dict:
    """Check a run and save the result beside it: check.json for version 1, check-v2.json for version 2 (each
    overwritten: it is worked out again from the run)."""
    result = check_run(folder, incidents_root, version)
    (Path(folder) / CHECK_FILES[version]).write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    return result


def is_run(folder: Path) -> bool:
    return all((folder / name).is_file() for name in (TRACE, TOOL_RESULTS, RUN_FILE))


def has_recording(folder: Path) -> bool:
    return (folder / RECORDING).is_file()
