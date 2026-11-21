"""Klaxon's diagnosis agent: a small slice of Klaxon, for the CSYE 7374 mini-project.

An alert fires on a synthetic payments system. The agent (an LLM in a plain-Python loop) chooses, one step at a time,
between three read-only tools (money_check, search_logs, recent_deploys), reads each result, and ends in a diagnosis
with evidence. Nothing is fixed automatically: the fix is only proposed. Code then checks the diagnosis against the
tool results the agent really got (diagnosis_agent.check), not against what the model claims.

Most of the agent is copied from Klaxon (github.com/AravindKumar2504/klaxon) at commit fef450b: each copied module
says so at its top, with what changed. New here: the three tools on captured incident data (tools), the cases (cases),
the evidence checks (check), the scripted mock model (scripted), and the command line (cli).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

__version__ = "0.1.0"
REPO = Path(__file__).resolve().parent.parent
PACKAGE = Path(__file__).resolve().parent
KLAXON_COMMIT = "fef450b0aae7545fc3f86ba8eb70ccce2dff1676"


def code_fingerprint() -> str:
    """The SHA-256 of this package's source, file by file in name order: a run records it, so a later reader knows
    exactly which code made it."""
    digest = hashlib.sha256()
    for path in sorted(PACKAGE.glob("*.py")):
        digest.update(path.name.encode() + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()
