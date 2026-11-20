"""Build the project document (docs/report.html) from this repo's own files: the incidents and the recorded runs.

Every number, alert, tool result and model reply in it is read from incidents/ and runs/ when it is built, so the
document cannot say more than the record holds. The model's words are quoted as recorded (Claude Opus 5.5 through
Claude Code, 2026-11-20 UTC). Run it, then scripts/render_pdf.py:

    .venv/bin/python scripts/build_report.py
    <a Python with Playwright> scripts/render_pdf.py docs/report.html docs/<name>.pdf

The contributions section is Aravind's own text, read from docs/contributions.md when that file exists.
"""

from __future__ import annotations

import html
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from diagnosis_agent import cases, check, cli  # noqa: E402
from diagnosis_agent.tools import DESCRIPTIONS  # noqa: E402

RUNS = ROOT / "runs"
OUT = ROOT / "docs" / "report.html"
SUCCESS = "2026-11-20T091900Z_poison-message_claude-opus-5-5"
MISSING_RUN = "2026-11-20T111408Z_missing-deploys_claude-opus-5-5"
SLIP_RUN = "2026-11-20T091947Z_missing-deploys_claude-opus-5-5"
REPO_URL = "https://github.com/AravindKumar2504/csye7374-klaxon-diagnosis-agent"
RECORDED = "Claude Opus 5.5 (claude-opus-5-5), as recorded on 2026-11-20 UTC"


def e(text: object) -> str:
    return html.escape(str(text), quote=False)


def records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def summary(tool: str, text: str, is_error: bool) -> str:
    """What one tool result held, in a line, from the result itself."""
    if is_error:
        return f"Tool error: {text} (simulated outage)" if "docker" in text else f"Tool error: {text}"
    body = json.loads(text)
    if tool == "money_check":
        if body.get("outcome") == "could not run":
            return f"Could not run: {body.get('reason')}."
        failed = [inv["name"] for inv in body.get("invariants", []) if not inv["passed"]]
        return f"Failed {len(failed)} of {len(body.get('invariants', []))} invariants: {', '.join(failed)}."
    if tool == "recent_deploys":
        newest = body["deploys"][0]
        settings = ", ".join(f"{k}={v}" for k, v in newest["settings"].items())
        return f"Newest: {newest['service']}, deployed {newest['deployed_at']} with {settings}."
    lines = body.get("lines") or []
    if not lines:
        return f"{body['matched']} lines matched."
    first = lines[0]
    fields = first.get("fields", {})
    shown = {
        k: fields[k] for k in ("event_id", "mode", "attempt", "error", "source", "reason", "offset") if k in fields
    }
    detail = "; ".join(f"{k} {v}" for k, v in shown.items())
    return (
        f'{body["matched"]} lines matched. Newest: {first["service"]} {first["level"]} "{first["message"]}" ({detail}).'
    )


def steps_of(folder: Path) -> tuple[list[dict], dict]:
    trace = records(folder / "trace.jsonl")
    calls = [r for r in records(folder / "tool-results.jsonl") if r.get("kind") == "tool_call"]
    rows, n = [], 0
    for step in trace:
        if step.get("kind") != "step":
            continue
        proposed = step.get("proposed") or {}
        if step.get("ran"):
            result = calls[n]
            n += 1
            rows.append(
                {
                    "step": step["step"],
                    "action": f"{step['ran']['call_id']} {step['ran']['tool']} {json.dumps(step['ran']['arguments'])}",
                    "why": proposed.get("why", ""),
                    "got": summary(result["tool"], result["text"], result["is_error"]),
                }
            )
        else:
            outcome = step.get("why", "")
            rows.append(
                {
                    "step": step["step"],
                    "action": proposed.get("action", "-"),
                    "why": proposed.get("why", ""),
                    "got": outcome,
                }
            )
    return rows, trace[-1]


def step_table(folder: Path) -> str:
    rows, _ = steps_of(folder)
    body = "".join(
        f"<tr><td>{r['step']}</td><td><code>{e(r['action'])}</code></td><td class='q'>{e(r['why'])}</td>"
        f"<td>{e(r['got'])}</td></tr>"
        for r in rows
    )
    return (
        "<table class='steps'><thead><tr><th>Step</th><th>What the model chose</th><th>Its reason, as recorded</th>"
        f"<th>What came back</th></tr></thead><tbody>{body}</tbody></table>"
    )


def checks_list(result: dict) -> str:
    items = "".join(
        f"<li><b>{'pass' if c['passed'] else 'FAIL'}</b> {e(c['check'])}: {e(c['detail'])}</li>"
        for c in result["checks"]
    )
    return f"<ul class='checks'>{items}</ul>"


def results_table() -> tuple[str, dict]:
    rows, totals = [], {"runs": 0, "calls": 0, "cost": 0.0}
    for folder in cli.run_folders([RUNS]):
        one, two = check.check_run(folder, version=1), check.check_run(folder, version=2)
        used = cli.usage(folder)
        totals["runs"] += 1
        totals["calls"] += used["replies"] + used["failed_calls"]
        totals["cost"] += used["cost_usd"]
        stop = one["stop_reason"]
        not_run = stop in cli.NOT_RUN
        verdict = one.get("cause", {}).get("verdict", "-")
        rows.append(
            "<tr>"
            f"<td>{e(folder.name[11:13])}:{e(folder.name[13:15])}:{e(folder.name[15:17])}</td><td>{e(one['case'])}</td>"
            f"<td>{'not run (seat limit)' if not_run else e(stop)}</td>"
            f"<td>{one['steps']}</td><td>{one['tool_calls']}</td>"
            f"<td>{e(verdict)}</td>"
            f"<td>{'-' if not_run else ('pass' if one['passed'] else 'fail')}</td>"
            f"<td>{'-' if not_run else ('pass' if two['passed'] else 'fail')}</td>"
            f"<td>{'-' if one['confidence'] is None else one['confidence']}</td>"
            f"<td>{one['seconds']:.0f}</td>"
            "</tr>"
        )
    head = (
        "<tr><th>Started (UTC)</th><th>Case</th><th>Stop</th><th>Steps</th><th>Tool calls</th><th>Cause (rule)</th>"
        "<th>Checks v1</th><th>Checks v2</th><th>Confidence</th><th>Seconds</th></tr>"
    )
    return f"<table class='results'><thead>{head}</thead><tbody>{''.join(rows)}</tbody></table>", totals


def alert_rows() -> str:
    out = []
    for case in cases.CASES.values():
        alert = case.alert()
        extra = " The deploy list answers with a docker timeout (simulated)." if case.withheld else ""
        out.append(
            f"<tr><td><b>{e(case.name)}</b><br>{e(case.title)}</td>"
            f"<td><b>{e(alert.name)}</b>: {e(alert.summary)}{e(extra)}</td><td>{e(case.right_answer)}</td></tr>"
        )
    return "".join(out)


DIAGRAM = """
<svg viewBox="0 0 700 300" class="diagram" role="img" aria-label="The agent loop">
  <defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">
  <path d="M0,0 L10,5 L0,10 z" fill="#334"/></marker></defs>
  <g font-family="Helvetica, Arial, sans-serif" font-size="12" fill="#111">
  <rect x="10" y="20" width="150" height="54" rx="8" fill="#fde7e7" stroke="#b33"/>
  <text x="85" y="42" text-anchor="middle" font-weight="bold">Alert</text>
  <text x="85" y="60" text-anchor="middle">from Klaxon's rules</text>
  <rect x="230" y="20" width="200" height="54" rx="8" fill="#e8eefc" stroke="#335"/>
  <text x="330" y="42" text-anchor="middle" font-weight="bold">Loop (plain Python)</text>
  <text x="330" y="60" text-anchor="middle">conversation, budgets, trace</text>
  <rect x="500" y="20" width="190" height="54" rx="8" fill="#eef7ee" stroke="#363"/>
  <text x="595" y="42" text-anchor="middle" font-weight="bold">Model: Claude Opus 5.5</text>
  <text x="595" y="60" text-anchor="middle">one JSON reply per step</text>
  <line x1="160" y1="47" x2="228" y2="47" stroke="#334" marker-end="url(#a)"/>
  <line x1="430" y1="38" x2="498" y2="38" stroke="#334" marker-end="url(#a)"/>
  <line x1="498" y1="54" x2="432" y2="54" stroke="#334" marker-end="url(#a)"/>
  <text x="464" y="32" text-anchor="middle" font-size="10">prompt</text>
  <text x="464" y="67" text-anchor="middle" font-size="10">reply</text>
  <rect x="230" y="120" width="200" height="54" rx="8" fill="#fff6dd" stroke="#a80"/>
  <text x="330" y="142" text-anchor="middle" font-weight="bold">call_tool</text>
  <text x="330" y="160" text-anchor="middle">allow-list, schema, budget</text>
  <rect x="500" y="120" width="190" height="54" rx="8" fill="#fff6dd" stroke="#a80"/>
  <text x="595" y="138" text-anchor="middle" font-weight="bold">One of 3 read-only tools</text>
  <text x="595" y="154" text-anchor="middle">money_check, search_logs,</text>
  <text x="595" y="168" text-anchor="middle">recent_deploys</text>
  <line x1="330" y1="74" x2="330" y2="118" stroke="#334" marker-end="url(#a)"/>
  <line x1="430" y1="147" x2="498" y2="147" stroke="#334" marker-end="url(#a)"/>
  <path d="M520,120 L520,92 L415,92 L415,76" fill="none" stroke="#334" marker-end="url(#a)"/>
  <text x="528" y="108" font-size="10">result, as untrusted data</text>
  <path d="M250,74 L250,100 L115,100 L115,228" fill="none" stroke="#334" marker-end="url(#a)"/>
  <text x="122" y="118" font-size="10">or: diagnose</text>
  <rect x="10" y="230" width="210" height="54" rx="8" fill="#e8eefc" stroke="#335"/>
  <text x="115" y="252" text-anchor="middle" font-weight="bold">Diagnosis</text>
  <text x="115" y="270" text-anchor="middle">validated, cites its calls</text>
  <rect x="250" y="230" width="210" height="54" rx="8" fill="#f1e8fc" stroke="#536"/>
  <text x="355" y="252" text-anchor="middle" font-weight="bold">Evidence checks (code)</text>
  <text x="355" y="270" text-anchor="middle">against the results it got</text>
  <rect x="490" y="230" width="200" height="54" rx="8" fill="#eee" stroke="#555"/>
  <text x="590" y="252" text-anchor="middle" font-weight="bold">A person decides</text>
  <text x="590" y="270" text-anchor="middle">the fix is never applied</text>
  <line x1="220" y1="257" x2="248" y2="257" stroke="#334" marker-end="url(#a)"/>
  <line x1="460" y1="257" x2="488" y2="257" stroke="#334" marker-end="url(#a)"/>
  </g>
</svg>
"""

CSS = """
@page { size: Letter; margin: 0.65in 0.7in 0.75in 0.7in; }
body { font-family: Helvetica, Arial, sans-serif; font-size: 10.2pt; line-height: 1.42; color: #111; }
h1 { font-size: 21pt; margin: 0 0 2pt; }
h2 { font-size: 13.5pt; margin: 16pt 0 5pt; border-bottom: 1px solid #ccc; padding-bottom: 2pt; break-after: avoid; }
h2 + p, h2 + table { break-before: avoid; }
h3 { font-size: 11pt; margin: 10pt 0 3pt; break-after: avoid; }
p, li { margin: 3pt 0; }
.sub { color: #444; margin: 0 0 10pt; }
code, pre { font-family: Menlo, Consolas, monospace; font-size: 8.3pt; }
pre { background: #f5f5f7; padding: 6pt 8pt; border-radius: 4px; white-space: pre-wrap; break-inside: avoid; }
table { border-collapse: collapse; width: 100%; margin: 5pt 0 8pt; font-size: 8.6pt; break-inside: auto; }
th, td { border: 1px solid #ccc; padding: 3pt 4pt; vertical-align: top; text-align: left; }
th { background: #f0f1f4; }
tr { break-inside: avoid; }
td.q { font-style: italic; }
.steps td:nth-child(2) { width: 27%; word-break: break-word; }
.steps td:nth-child(3) { width: 33%; }
.diagram { width: 100%; height: auto; margin: 4pt 0 6pt; }
.quote { border-left: 3px solid #889; padding: 2pt 8pt; margin: 5pt 0; background: #fafafc; font-size: 9.4pt; }
.label { font-size: 9pt; color: #444; margin-top: 6pt; }
.checks { font-size: 9pt; }
.todo { background: #fff3c4; border: 1px dashed #b80; padding: 6pt; }
"""


def main() -> int:
    results, totals = results_table()
    success_dir, missing_dir, slip_dir = RUNS / SUCCESS, RUNS / MISSING_RUN, RUNS / SLIP_RUN
    _, success_stop = steps_of(success_dir)
    _, missing_stop = steps_of(missing_dir)
    success_check = check.check_run(success_dir, version=1)
    missing_check = check.check_run(missing_dir, version=2)
    slip = check.check_run(slip_dir, version=2)
    slip_item = next(
        i
        for i in next(c for c in slip["checks"] if c["check"] == "cited_results_hold_the_facts")["evidence"]
        if i["not_in_that_result"]
    )
    slip_shows = next(
        x["shows"]
        for x in records(slip_dir / "trace.jsonl")[-1]["diagnosis"]["evidence"]
        if x["call_id"] == slip_item["call_id"]
    )
    contributions_file = ROOT / "docs" / "contributions.md"
    if contributions_file.exists():
        paragraphs = [
            p.strip() for p in contributions_file.read_text().split("\n\n") if p.strip() and not p.startswith("#")
        ]
        contributions = "".join(f"<p>{e(p)}</p>" for p in paragraphs)
    else:
        contributions = "<p class='todo'>[Aravind writes this section, in his own words.]</p>"
    tools_rows = "".join(
        f"<tr><td><code>{e(name)}</code></td><td>{e(DESCRIPTIONS[name])}</td></tr>"
        for name in ("money_check", "search_logs", "recent_deploys")
    )
    page = f"""<!doctype html><html><head><meta charset="utf-8"><title>Klaxon Diagnosis Agent</title>
<style>{CSS}</style></head><body>
<h1>Klaxon Diagnosis Agent</h1>
<p class="sub">A payment incident diagnosis agent. CSYE 7374 Introduction to Agentic AI, Fall 2026, mini-project.<br>
Aravind Sundaravadivelu. Code, data and every recorded run: <a href="{REPO_URL}">{REPO_URL}</a></p>

<h2>1. The problem, and why it is useful</h2>
<p>A payment system can fail without going down. In this project's first incident, nothing crashed. Yet the ledger had booked
five events more than once, and our books were $116.97 ahead of the payment provider's records: a deploy had removed the
ledger's dedupe, 30 seconds before the alert. An on-call engineer has to find causes like this fast, from the
money, the logs and what changed. They must also say what the evidence leaves open, instead of guessing.</p>
<p>This agent does the first part of that job. An alert fires. An LLM investigates it with three read-only tools, one step at a
time, and ends with a diagnosis: the cause, the evidence (citing its own tool calls), a confidence, and a proposed fix.
Nothing is fixed automatically: a person decides. Then code checks the diagnosis against the tool results the run actually
received, not against what the model says. It is a small slice of Klaxon, Aravind's larger project
(<a href="https://github.com/AravindKumar2504/klaxon">github.com/AravindKumar2504/klaxon</a>), an AI on-call engineer for payments.</p>

<h2>2. The approved scope</h2>
<p>The professor approved the project with a reduced scope (announcement of 2026-11-02): three diagnostic tools, two
simulated payment incidents, one case with missing evidence, and no automatic fixes. This project is exactly that:</p>
<table><tr><th>Scope</th><th>Here</th></tr>
<tr><td>3 diagnostic tools</td><td><code>money_check</code>, <code>search_logs</code>, <code>recent_deploys</code></td></tr>
<tr><td>2 simulated payment incidents</td><td><code>duplicate-delivery</code> (payments counted twice) and <code>poison-message</code> (the ledger stops booking)</td></tr>
<tr><td>1 case with missing evidence</td><td><code>missing-deploys</code>: incident 1 again, but the deploy list does not answer</td></tr>
<tr><td>No automatic fixes</td><td>The agent has no tool that writes. Its fix is a proposal, recorded as not applied.</td></tr></table>

<h2>3. The system, the incidents and the data</h2>
<p>The incidents ran on Klaxon's synthetic payments system, on Docker Compose: a payments API that calls a fake payment
provider; the provider's signed webhooks, which a receiver puts on Kafka; a ledger consumer that books each event in a
double-entry ledger (PostgreSQL); a payment confirmer; Redis; and a money check that compares our books with the provider's
own records. All data is synthetic: no real customers or money.</p>
<table><tr><th>Case</th><th>The alert the agent got (raised by Klaxon's own rules)</th><th>Right answer (written before any run)</th></tr>
{alert_rows()}</table>
<p><b>How the data was made.</b> <code>scripts/capture_incident.py</code> drove Klaxon's own fault runner on its live stack (Klaxon
at commit <code>fef450b</code>), on 2026-11-20 UTC: synthetic traffic (3 payments a second), a warm-up, the fault injected, then
Klaxon's alert rules until one fired. At that moment, with the fault still in place, the script called Klaxon's own read-only
tools and saved what came back: the money check, the deploy list, and every log line (2,467 for incident 1, 6,314 for
incident 2). The agent's three tools answer from that saved data, so anyone can run the agent with Python alone, and every run
sees the same incident. For incident 2, the warm-up was 180 s, not Klaxon's 60 s: two tries at 60 s stopped before any
injection, because payments were already slow on the fresh stack (recorded in its <code>incident.json</code>).</p>

<h2>4. How the agent works, step by step</h2>
{DIAGRAM}
<p><b>The loop.</b> The loop sends the model the alert, the rules, the reply format (as JSON Schema) and the three tools. Each
reply is one step and one JSON object: either call one tool, with a reason, or give the diagnosis. The loop validates every
reply. An invalid one goes back with what was wrong, at most twice in a row. A tool call runs only if the tool is on the
read-only allow-list and the budget allows it (10 steps, 8 tool calls, 600 s). Its result goes back into the conversation
inside a JSON document under a fixed note that it is data, never instructions. Then the model decides the next step. The run
stops at a valid diagnosis or a budget. It must cite tool calls that ran (c1, c2, ...).</p>
<p><b>The model's decisions.</b> At every step the model chooses: which tool, with which arguments, or whether it has enough to
diagnose. Nothing in the loop fixes the order. The diagnosis holds a hypothesis, evidence with call ids, a confidence from 0
to 1, a proposed fix (its kind, target and summary), the fix's blast radius, and a rollback plan.</p>
<p><b>The tools it chooses between</b> (descriptions as the model sees them):</p>
<table><tr><th>Tool</th><th>What it shows</th></tr>{tools_rows}</table>
<p><b>How results drive the next step.</b> Every run that got a reply started with the money check, to confirm the alert.
What came back decided what came next. When the check showed money counted twice (incident 1), it read the deploy list next.
When the check could not settle and named the ledger as behind (incident 2), it read the ledger consumer's error logs first.
Its later searches used the exact event ids, Kafka offsets, trace ids and payment ids that earlier results had named.
Sections 5 and 6 show two runs step by step.</p>
<p><b>Guardrails.</b> Tools are read-only, checked against an allow-list when the tools are listed and again at each call.
Replies are checked against the schema, with bounded retries. Every tool result is framed as untrusted data. Live runs go
through Claude Code with no tools and no MCP servers of its own, checked at every start. Every reply and every tool result is
recorded before use, so any run replays offline, exactly.</p>

<h2>5. A run, step by step: incident 2</h2>
<p>Run <code>{e(SUCCESS)}</code>. The model's reasons are its own words: {RECORDED}. "What came back" is summarized by code from
each recorded result.</p>
{step_table(success_dir)}
<p class="label">The diagnosis it gave (hypothesis, as recorded):</p>
<div class="quote">{e(success_stop["diagnosis"]["hypothesis"])}</div>
<p>The checks, version 1 (written before the runs):</p>{checks_list(success_check)}

<h2>6. The problem case: missing evidence</h2>
<p>In <code>missing-deploys</code>, the deploy list answers every call with the error Klaxon's tool gives when docker does not
answer. Run <code>{e(MISSING_RUN)}</code> ({RECORDED}):</p>
{step_table(missing_dir)}
<p class="label">The diagnosis (hypothesis, as recorded):</p>
<div class="quote">{e(missing_stop["diagnosis"]["hypothesis"])}</div>
<p>It did not name the deploy's setting from the list it never got. Instead it found the ledger consumer restarting in
<code>mode=naive</code> in the consumer's own logs, and said the setting change was unconfirmed. Version 1 of the checks failed
this run on one point: its proposed fix named <code>LEDGER_HANDLER=idempotent</code>, which no result states. Version 2
reads the fix as a proposal and passes it:</p>
{checks_list(missing_check)}

<h2>7. Testing and results</h2>
<p>Nine live runs, three per case, plus one run the seat's 5-hour limit stopped at its first call (kept as recorded, counted
as not run, and run again after the reset). In all {totals["calls"]} model calls, ${totals["cost"]:.2f} at API prices by Claude
Code's own estimate (the runs used Aravind's university seat, which is not billed per call).</p>
{results}
<p><b>The cause.</b> Klaxon's rule (rubric version 4, copied unchanged and checked against Klaxon's own) says all 9 runs name the
recorded cause. It reads words, not meaning. In Klaxon, it agreed with Aravind's own blind verdicts on 27 of the 40 runs he
labeled: 13 of 13 in its calibration, but less on the hardest cases, and Klaxon's records call the graders a weak guide on
missing deploys. So the evidence checks below matter more than the rule's verdict.</p>
<p><b>Results checked against the evidence.</b> Version 1 of the checks was committed before the runs. It passed 3 of 9. Four
runs stated a total they had worked out themselves: "$134.47", the four payments counted twice. The sum is right (16.97 +
19.29 + 66.38 + 31.83), but no result states it, and version 1 only finds facts that a result states. One named a setting only
in its proposed fix. One cited the wrong call: "{e(slip_shows)}" cites <code>{e(slip_item["call_id"])}</code>, whose result does not hold
<code>{e(", ".join(slip_item["not_in_that_result"]))}</code> (the money check's result does). Version 2 was written after reading those
results, and is labeled so: it also accepts a sum of the money check's own findings, and reads the proposed fix as a proposal.
It passes 8 of 9. The citation slip still fails, as it should.</p>
<p><b>Problem paths, live.</b> In 6 of the 9 runs the model's first diagnosis broke the format: its hypothesis ran over the
1,000-character limit. The loop sent it back with that reason, and each time the model shortened it on the next step. In the
missing-evidence runs, the deploy list failed every time it was called (4 calls in 3 runs). One run asked for it twice.</p>
<p><b>Offline tests.</b> {cli_count()} tests, with no model and no Docker: the three tools on hand-written fixtures and on the
captured data; the loop's problem paths (a reply that is not JSON, a tool not offered, bad arguments, a citation of a call that
never ran, three invalid replies in a row, the tool budget, a model that times out, text in a log line that tries to give the
model orders); every check, passing and failing; the copied rule; the Claude Code adapter against a fake program; and an exact
replay of every live run.</p>
<p><b>Limits.</b> Three runs per case show the agent working, not how often it works. The tools read saved data, not a live
system. The checks match facts as text: they do not check counts, and the cause check reads words. One model was tested.</p>

<h2>8. How to run it</h2>
<pre># set up (Python 3.12)
python3.12 -m venv .venv
.venv/bin/pip install -e '.[dev]'
# every test, offline
.venv/bin/pytest -q
# the three cases and their alerts
.venv/bin/python -m diagnosis_agent cases
# one offline run with the scripted mock model
.venv/bin/python -m diagnosis_agent run missing-deploys --model scripted
# replay a recorded live run, offline, and check it reproduces
.venv/bin/python -m diagnosis_agent replay \\
    runs/{SUCCESS}
# every recorded run, with both versions of the checks
.venv/bin/python -m diagnosis_agent report
# one live run: needs Claude Code, signed in
.venv/bin/python -m diagnosis_agent run poison-message --model claude-opus-5-5</pre>
<p>The agent needs only Python 3.12's standard library; the tests need pytest. <code>--model scripted</code> is a mock whose
replies are worked out by code in a fixed order: it shows the pipeline offline, never how a model chooses.</p>

<h2>9. Contributions and AI use</h2>
{contributions}
<p class="label">Most of the agent's code is copied from Klaxon at commit fef450b, and each copied file says so at its top. Every
model reply in this document is one actually received, saved with its date and model.</p>
</body></html>"""
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(page)
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


def cli_count() -> int:
    """How many offline tests there are: collected, not run (pytest --collect-only)."""
    import subprocess

    done = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q"], cwd=ROOT, capture_output=True, text=True, timeout=120
    )
    last = [line for line in done.stdout.splitlines() if "tests collected" in line or "test collected" in line]
    return int(last[-1].split()[0]) if last else 0


if __name__ == "__main__":
    raise SystemExit(main())
