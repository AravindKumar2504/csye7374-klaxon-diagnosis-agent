# Klaxon diagnosis agent

A small AI agent that diagnoses payment incidents. It is Aravind Sundaravadivelu's mini-project for CSYE 7374 (Introduction to Agentic AI, Fall 2026), approved as "Klaxon, Payment Incident Diagnosis Agent", and a slice of his larger project, [Klaxon](https://github.com/AravindKumar2504/klaxon).

An alert fires on a synthetic payments system. The agent, an LLM in a plain-Python loop, picks one of three read-only tools at each step, reads the result, and decides what to look at next. It stops with a diagnosis: the cause, the evidence (citing its own tool calls), a confidence, and a proposed fix. Nothing is fixed automatically. Then code checks the diagnosis against the tool results the run actually received, not against what the model says.

## The approved scope

The professor approved the project with a reduced scope (announcement of 2026-11-02): three diagnostic tools, two simulated payment incidents, one case with missing evidence, and no automatic fixes.

| Scope | Here |
|---|---|
| 3 diagnostic tools | `money_check` (reconcile our books with the provider's records), `search_logs` (the four services' logs), `recent_deploys` (what changed, and when) |
| 2 simulated payment incidents | `duplicate-delivery`: payments counted twice in the books (Klaxon's fault 1). `poison-message`: the ledger stops booking, stuck on one bad message (Klaxon's fault 2) |
| 1 case with missing evidence | `missing-deploys`: incident 1 again, but the deploy list does not answer (a simulated docker outage) |
| No automatic fixes | The agent has no write tool. The fix is only proposed, and the trace records it as not applied |

## How it works

```
alert ──> [ model: pick one action ] ──> call a tool ──> result, framed as untrusted data ──┐
              ^                                                                             │
              └──────────────── the conversation so far, and the budget left ◄──────────────┘
          or: diagnose ──> the loop validates it ──> the run stops ──> code checks it against the evidence
```

1. The loop sends the model the alert, the rules, the reply format (JSON Schema) and the three tools.
2. Each reply is one JSON object: call one tool, or give the diagnosis. The loop checks it against the schema. An invalid reply goes back with what was wrong, at most twice in a row.
3. A tool call runs only if the tool is on the read-only allow-list and the budget allows it (10 steps, 8 tool calls, 600 s). Its result goes back to the model inside a JSON document under a fixed note that it is data, never instructions.
4. The diagnosis must cite calls that ran (c1, c2, ...). The run then stops, and `diagnosis_agent.check` compares it with the evidence.

Every step goes to `trace.jsonl`. A live run also records every model reply (`model-recording.jsonl`) and every tool result (`tool-results.jsonl`) before the loop uses them, so any live run replays offline, exactly.

## Run it

You need Python 3.12 on macOS or Linux. Nothing else for the offline parts. Live runs also need [Claude Code](https://claude.com/claude-code), signed in.

```bash
python3.12 -m venv .venv
.venv/bin/pip install -e '.[dev]'
```

Offline, with no model:

```bash
.venv/bin/python -m diagnosis_agent cases
.venv/bin/python -m diagnosis_agent prompt duplicate-delivery
.venv/bin/python -m diagnosis_agent tool duplicate-delivery search_logs '{"service": "ledger-consumer", "level": "ERROR"}'
.venv/bin/python -m diagnosis_agent run missing-deploys --model scripted
.venv/bin/python -m diagnosis_agent report
.venv/bin/pytest -q
```

`--model scripted` is a mock: its replies are worked out by a few lines of Python, always in the same order. It shows the loop, tools, trace and checks working offline. It says nothing about how a model chooses. Its runs go to `runs-scripted/`, which git ignores.

Replay a recorded live run, offline, and check that it reproduces:

```bash
.venv/bin/python -m diagnosis_agent replay runs/2026-11-20T091900Z_poison-message_claude-opus-5-5
.venv/bin/python -m diagnosis_agent check runs/2026-11-20T091900Z_poison-message_claude-opus-5-5
```

Live, on a Claude Code login (the live runs below made 5 to 10 model calls each):

```bash
.venv/bin/python -m diagnosis_agent run duplicate-delivery --model claude-opus-5-5
```

## Results

Nine live runs on 2026-11-20 UTC: Claude Opus 5.5 (`claude-opus-5-5`) through Claude Code 2.1.326, three per case, on the captured incidents. Every reply and tool result is in `runs/`, and every run replays offline, exactly (`tests/test_live_runs.py`). In all, 73 model calls, $3.58 at API prices by Claude Code's own estimate (the runs used Aravind's seat, which is not billed per call).

| Case | Runs | Names the cause (Klaxon's rule) | Every check passed, version 1 | Every check passed, version 2 |
|---|---|---|---|---|
| duplicate-delivery | 3 | 3 of 3 | 0 of 3 | 3 of 3 |
| poison-message | 3 | 3 of 3 | 3 of 3 | 3 of 3 |
| missing-deploys | 3 | 3 of 3 | 0 of 3 | 2 of 3 |

- **The agent chose its own path.** Every run started with the money check, to confirm the alert. What came back decided the next step. When the money was wrong (incident 1), it read the deploy list next. When the books could not settle (incident 2), it read the ledger consumer's error logs first. It then searched the logs for the event ids, Kafka offsets and trace ids that earlier results named. Runs took 4 to 8 tool calls and 25 to 65 seconds.
- **The missing-evidence case.** When `recent_deploys` failed, each run said the deploy list could not be read, and none named the deploy's setting from that list. One of them (round 2) found the ledger consumer restarting in `mode=naive` in its own logs instead. One run asked for the deploy list a second time, and it failed again.
- **Version 1 of the checks** was committed before the runs (each run's `check.json`). It passed 3 of 9. Four runs stated a total they worked out themselves ($134.47, the four payments counted twice). It is right, but no result states it. One named a setting only in its proposed fix. One cited `c8` for `evt_000288`, which is in the result of `c1`, not `c8`.
- **Version 2** was written after reading those results, and is labeled so (each run's `check-v2.json`). It also accepts a sum of the money check's own findings, and reads the proposed fix as a proposal, not as a claim. It passes 8 of 9. The citation slip still fails.
- One more run, the second of incident 2, stopped at its first call: the seat's 5-hour limit. It is kept as recorded and counted as not run, and the case was run again after the reset.

`.venv/bin/python -m diagnosis_agent report --markdown` prints every run's row.

## Sample data

`incidents/` holds the two incidents, captured from Klaxon's synthetic payments system running live on Docker Compose (Klaxon at commit `fef450b`), by `scripts/capture_incident.py`. Klaxon's own fault runner injected each fault under synthetic traffic. Klaxon's own alert rules raised the alert. At that moment the script called Klaxon's own read-only tools in place of a model and saved what came back: the money check, the deploy list, and every log line the services had sent. All data is synthetic: no real customers, payments or money. Each folder's `incident.json` says when and how it was captured, and `harness.json` holds Klaxon's record of the fault run.

## What came from Klaxon

Most of the agent is copied from Klaxon at commit `fef450b`. Each copied file says so at its top, with what changed.

| File | From Klaxon | Changes |
|---|---|---|
| `diagnosis_agent/loop.py` | `src/klaxon/agent/loop.py` | cut to the plain loop: three tools, no write actions, no LangGraph, no MCP process |
| `diagnosis_agent/contract.py` | `src/klaxon/agent/contract.py` | no optional write action in the diagnosis |
| `diagnosis_agent/model.py` | `src/klaxon/agent/model.py` | two docstring lines |
| `diagnosis_agent/claude_code.py` | `src/klaxon/agent/claude_code.py` | provenance names this repo |
| `diagnosis_agent/trace.py` | `src/klaxon/agent/trace.py` | imports |
| `diagnosis_agent/tool_client.py` | `src/klaxon/agent/mcp_client.py` | the tool interface and record/replay only, no MCP transport |
| `diagnosis_agent/text.py` | `src/klaxon/agent/tools.py` | only its cleaning and argument checks |
| `diagnosis_agent/grading.py` | `src/klaxon/grading.py` | rubric version 4 for these cases, patterns unchanged (`scripts/verify_copied_rubric.py` checks them against Klaxon's) |

New here: `tools.py` (the three tools on the captured data), `cases.py`, `check.py`, `scripted.py`, `cli.py`, the tests, and the two scripts.

## Documents

- `docs/Klaxon_Diagnosis_Agent_Aravind_Sundaravadivelu.pdf`: the project document, built from the incidents and the recorded runs by `scripts/build_report.py` (source: `docs/report.html`).
- `docs/learned.md` and `docs/contributions.md`: Aravind's own words, as they appear in the class deck and the PDF.

## AI use

Claude (through Claude Code) wrote most of the code and docs, in Klaxon and here, under Aravind's direction. He chose the project, the scope, the three tools, the two incidents and the missing-evidence case, how Klaxon is reused, the model, and the number of live runs, and he reviewed the result. Every model reply shown anywhere in this repo is one actually received, saved with its date and model.
