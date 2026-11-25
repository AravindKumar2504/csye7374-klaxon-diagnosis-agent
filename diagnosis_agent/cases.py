"""The three cases of the approved scope: two simulated payment incidents, and one case with missing evidence.

New for this mini-project, from Klaxon at fef450b: the causes and remedies are the ones Klaxon's faults record
(src/klaxon/faults.py), and the missing-evidence case is Klaxon's missing-deploys case (src/klaxon/evaluation.py) with
only this agent's one docker-backed tool withheld. Each right answer is written here before any live run on it.

    duplicate-delivery   incident 1: payments counted twice in the books (Klaxon's fault 1)
    poison-message       incident 2: the ledger stops booking, stuck on one message (Klaxon's fault 2)
    missing-deploys      incident 1's data again, with the deploy list unavailable (a simulated docker outage)

The alert of each case is the one Klaxon's own alert rules raised when the incident was captured (incidents/<name>/
incident.json). The budgets are the ones Klaxon's evaluation runs had: 10 steps and 8 tool calls.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from diagnosis_agent import REPO
from diagnosis_agent.loop import Alert, Budgets
from diagnosis_agent.tools import DOCKER_SILENT, Incident, IncidentTools, Withheld

INCIDENTS = REPO / "incidents"
INCIDENT, MISSING = "incident", "missing evidence"
BUDGETS = Budgets(steps=10, tool_calls=8, seconds=600.0)  # Klaxon's evaluation budgets (STEPS, TOOL_CALLS)


@dataclass(frozen=True)
class Case:
    name: str
    kind: str  # INCIDENT or MISSING
    title: str
    incident: str  # the folder under incidents/ whose data the tools read
    rubric: str  # the cause rubric (diagnosis_agent.grading.RUBRICS) its hypothesis is graded by
    right_answer: str
    remedy: str
    withheld: dict[str, str] = field(default_factory=dict)  # tool name -> the error it answers with instead

    def folder(self, root: Path = INCIDENTS) -> Path:
        return root / self.incident

    def alert(self, root: Path = INCIDENTS) -> Alert:
        """The alert Klaxon's rules raised when the incident was captured, exactly as saved."""
        return Alert(**Incident(self.folder(root)).about["alert"])

    def tools(self, root: Path = INCIDENTS) -> IncidentTools | Withheld:
        tools = IncidentTools(self.folder(root))
        return Withheld(tools, self.withheld) if self.withheld else tools

    def describe(self) -> dict:
        return {
            "name": self.name,
            "kind": self.kind,
            "title": self.title,
            "incident": self.incident,
            "rubric": self.rubric,
            "right_answer": self.right_answer,
            "remedy": self.remedy,
            "withheld": dict(self.withheld),
        }


FAULT_1_CAUSE = (
    "A deploy of the ledger consumer without its dedupe (LEDGER_HANDLER=naive): every webhook the provider sends again,"
    " after an acknowledgement it never got, is booked a second time."
)
FAULT_1_REMEDY = (
    "Roll the ledger consumer back to the idempotent handler, then repair each double posting with a reversal (a"
    " write, so it needs a person's approval)."
)

CASES = {
    case.name: case
    for case in (
        Case(
            name="duplicate-delivery",
            kind=INCIDENT,
            title="Payments counted twice in the books",
            incident="duplicate-delivery",
            rubric="duplicate-delivery",
            right_answer=FAULT_1_CAUSE,
            remedy=FAULT_1_REMEDY,
        ),
        Case(
            name="poison-message",
            kind=INCIDENT,
            title="The ledger stops booking: a poison message",
            incident="poison-message",
            rubric="poison-message",
            right_answer=(
                "A deploy of the ledger consumer that retries unreadable messages forever (BAD_EVENTS=retry), then one"
                " unreadable event from the provider (a payment.updated with its amount as a decimal string): the"
                " ledger consumer is stuck on it, and books nothing behind it."
            ),
            remedy=(
                "Roll the ledger consumer back to dead-lettering: the poison goes to the dead-letter topic, the backlog"
                " drains, and the money check passes again."
            ),
        ),
        Case(
            name="missing-deploys",
            kind=MISSING,
            title="Payments counted twice in the books, with the deploy list unavailable",
            incident="duplicate-delivery",
            rubric="duplicate-delivery",
            right_answer=(
                f"{FAULT_1_CAUSE} Docker did not answer the tool that reads it, so the deploy list could not be read:"
                " a right diagnosis says what it could not check, and claims nothing it has no evidence for, such"
                " as which deploy brought the change, or when."
            ),
            remedy=FAULT_1_REMEDY,
            withheld={"recent_deploys": DOCKER_SILENT},
        ),
    )
}


def get(name: str) -> Case:
    if name not in CASES:
        raise ValueError(f"no case {name!r}; the cases are: {', '.join(CASES)}")
    return CASES[name]
