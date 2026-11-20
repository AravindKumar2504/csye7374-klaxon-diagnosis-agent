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
.venv/bin/python -m diagnosis_agent replay runs/<a run's folder>
.venv/bin/python -m diagnosis_agent check runs/<a run's folder>
```

Live, on a Claude Code login (it makes about 5 to 8 model calls per run):

```bash
.venv/bin/python -m diagnosis_agent run duplicate-delivery --model claude-opus-5-5
```

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

## AI use

Claude (through Claude Code) wrote most of the code and docs, in Klaxon and here, under Aravind's direction. He chose the project, the scope, the three tools, the two incidents and the missing-evidence case, how Klaxon is reused, the model, and the number of live runs, and he reviewed the result. Every model reply shown anywhere in this repo is one actually received, saved with its date and model.
