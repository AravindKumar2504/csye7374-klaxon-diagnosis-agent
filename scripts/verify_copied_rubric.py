"""Check that diagnosis_agent.grading holds Klaxon's rubric version 4 unchanged, and gives Klaxon's verdicts.

Run it with Klaxon's own venv, from a Klaxon checkout at fef450b (it imports both):

    /path/to/klaxon/.venv/bin/python scripts/verify_copied_rubric.py

It compares every element and pattern of the two rubrics this repo copies (duplicate-delivery and poison-message) and
the missing-deploys case's could-not-check element with Klaxon's, then grades sample sentences with both and compares
the verdicts. It prints what it found and exits 1 on any difference.
"""

from __future__ import annotations

import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from klaxon import grading as klaxon  # noqa: E402

from diagnosis_agent import grading as ours  # noqa: E402

SAMPLES = (
    "The ledger consumer books redelivered webhooks twice: its dedupe is gone after a deploy.",
    "Redis lost its idempotency keys, so the ledger posted events twice.",
    "The ledger consumer is stuck retrying one malformed message forever.",
    "A database outage makes the ledger retry.",
    "Payments are slow.",
    "The ledger consumer's duplicate check was turned off, so retried webhooks are posted again.",
    "The ledger consumer cannot book an event whose amount is not a positive integer, and retries it forever.",
)


def main() -> int:
    problems = []
    for name in ("duplicate-delivery", "poison-message"):
        theirs = asdict(klaxon.RUBRICS_V4[name])
        theirs.pop("also")
        if theirs != asdict(ours.RUBRICS[name]):
            problems.append(f"{name}: the rubric differs from Klaxon's")
    if asdict(klaxon.RUBRICS_V4["missing-deploys"].also[0]) != asdict(ours.COULD_NOT_CHECK):
        problems.append("missing-deploys: the could-not-check element differs from Klaxon's")
    fix = {"kind": "rollback", "target": "ledger-consumer", "summary": "roll it back"}
    compared = 0
    for sentence in SAMPLES:
        for name in ("duplicate-delivery", "poison-message"):
            theirs = klaxon.compare(name, {"hypothesis": sentence, "proposed_fix": fix})["verdict"]
            mine = ours.compare(name, sentence, fix)["verdict"]
            compared += 1
            if theirs != mine:
                problems.append(f"{name}: {sentence!r}: Klaxon says {theirs}, this repo says {mine}")
    print(f"Compared 2 rubrics, 1 element and {compared} verdicts with Klaxon's (rubric version 4).")
    print("\n".join(problems) if problems else "All the same.")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
