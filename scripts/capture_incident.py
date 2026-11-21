"""Capture one incident from Klaxon's live stack, as the data this repo's three tools answer from.

Run it from a Klaxon checkout at commit fef450b, with that checkout's own .venv and its stack up (make up there), and
with KLAXON_DEPLOY_LOG naming a file of its own, so that Klaxon's real deploy log (~/.klaxon/deploys.jsonl) is never
touched:

    KLAXON_DEPLOY_LOG=/path/to/capture-deploys.jsonl /path/to/klaxon/.venv/bin/python scripts/capture_incident.py \\
        duplicate-delivery --out incidents/duplicate-delivery

It drives Klaxon's own fault runner (klaxon.fault_runs.run), unchanged: synthetic traffic (3 payments a second, seed 7,
as in Klaxon's fault runs), a minute's warm-up, two quiet evaluations of the alert rules, the fault injected, then the
rules until one fires. At that moment, with the fault in place and the traffic flowing, this script calls Klaxon's own
read-only tools in place of a model: money_check (with its defaults: it first waits up to 60 s to settle), then
recent_deploys, then it reads every line Loki holds for the four services. Then the runner clears the fault, reads the
rules once more after 90 s, and stops the traffic, as in every Klaxon fault run.

It writes, into --out:
    incident.json         what was captured, when, from which commit and command; the fault and the alert
    money_check.json      Klaxon's money_check result, as its tool returned it
    recent_deploys.json   Klaxon's recent_deploys result, as its tool returned it
    logs.jsonl            every log line, oldest first, in the format of Klaxon's search_logs (one per line)
    harness.json          Klaxon's fault-run record: every evaluation of the rules, the timings, the traffic

This script imports Klaxon, so it runs only with Klaxon's venv. Nothing else in this repo imports Klaxon.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from klaxon import alerts, fault_runs, faults, traffic
from klaxon.agent import tools as klaxon_tools
from klaxon.agent.loop import RunResult
from klaxon.patient.stack import REPO

KLAXON_COMMIT = "fef450b0aae7545fc3f86ba8eb70ccce2dff1676"
SERVICES = ("payments-api", "webhook-receiver", "ledger-consumer", "payment-confirmer")
LOKI_LIMIT = 5000  # Loki's own cap on lines per query; the read pages past it
LABEL = (
    "Captured from Klaxon's synthetic payments system, live on Docker Compose on this Mac: synthetic data only, no"
    " real customers or money. Klaxon's own fault runner injected the fault under synthetic traffic, and Klaxon's own"
    " alert rules raised the alert from what the stack showed. At that moment, with the fault in place, this script"
    " called Klaxon's own read-only tools in place of a model: money_check and recent_deploys, as their results came"
    " back, and every log line Loki held for the four services. No model was called."
)


def iso(seconds: float) -> str:
    return datetime.fromtimestamp(seconds, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def git(*args: str) -> str | None:
    done = subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, timeout=10)
    return done.stdout.strip() if done.returncode == 0 else None


THIS_REPO = Path(__file__).resolve().parent.parent


def shown(argument: str) -> str:
    """An argument as the saved command shows it: a path in this repo relative to it, a temporary folder as <tmp>,
    and this Mac's home folder as ~ (as Klaxon's evidence writes it), so no local path or user name is saved."""
    for root, label in ((str(THIS_REPO) + "/", ""), ("/private/tmp/", "<tmp>/"), ("/tmp/", "<tmp>/")):
        if argument.startswith(root):
            rest = argument[len(root) :]
            return label + (rest if not label else Path(rest).name)
    return argument.replace(str(Path.home()), "~")


def read_logs(sources: klaxon_tools.Sources, start: float, end: float) -> list[dict]:
    """Every line Loki holds for the four services from start to end (seconds), oldest first, each as Klaxon's
    search_logs gives a line. A query that comes back full is read again from just after its last line."""
    seen: set[tuple] = set()
    lines: list[tuple[int, dict]] = []
    for service in SERVICES:
        cursor = int(start * 1e9)
        while True:
            params = {
                "query": f'{{service_name="{service}"}}',
                "start": cursor,
                "end": int(end * 1e9),
                "limit": LOKI_LIMIT,
                "direction": "forward",
            }
            data = klaxon_tools._get_json(sources, sources.loki_url, "/loki/api/v1/query_range", params, "Loki")
            result = (data.get("data") or {}).get("result") or []
            page = sorted(
                (
                    (int(stamp), stream.get("stream") or {}, text)
                    for stream in result
                    for stamp, text in stream.get("values") or []
                ),
                key=lambda line: line[0],
            )
            for stamp, labels, text in page:
                key = (stamp, json.dumps(labels, sort_keys=True), text)
                if key not in seen:
                    seen.add(key)
                    lines.append((stamp, klaxon_tools._log_line(stamp, labels, text)))
            if len(page) < LOKI_LIMIT:
                break
            cursor = page[-1][0] + 1
    lines.sort(key=lambda line: line[0])
    return [line for _, line in lines]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("fault", choices=("duplicate-delivery", "poison-message"))
    parser.add_argument("--out", type=Path, required=True, help="the incident folder to write (must not exist)")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--rate", type=float, default=3.0)
    parser.add_argument("--logs-since-minutes", type=float, default=15.0, help="read logs from this long before now")
    parser.add_argument(
        "--warmup-seconds",
        type=float,
        default=fault_runs.WARMUP_SECONDS,
        help=f"traffic before the rules are first read (Klaxon's default: {fault_runs.WARMUP_SECONDS:g} s)",
    )
    parser.add_argument("--note", default="", help="a sentence for incident.json: why a setting differs, say")
    parser.add_argument(
        "--record-on-failure", type=Path, help="where to save the runner's record if nothing is captured"
    )
    args = parser.parse_args(argv)

    if not os.environ.get("KLAXON_DEPLOY_LOG"):
        print("Refused: set KLAXON_DEPLOY_LOG to a file of its own, so Klaxon's real deploy log is not touched.")
        return 2
    commit = git("rev-parse", "HEAD")
    if commit != KLAXON_COMMIT:
        print(f"Refused: Klaxon at {REPO} is at {commit}, not {KLAXON_COMMIT[:7]}.")
        return 2
    if args.out.exists():
        print(f"Refused: {args.out} exists; captures are never overwritten.")
        return 2

    fault = faults.get(args.fault)
    settings = traffic.Settings(rate=args.rate, seed=args.seed)
    stack = fault_runs.Stack(faults.LocalStack(), alerts.Alerting())
    record = fault_runs.Record(fault.describe(), asdict(settings), {})
    sources = klaxon_tools.Sources()
    tools = klaxon_tools.make_tools(sources)
    logs_from = time.time() - args.logs_since_minutes * 60
    captured: dict = {}

    def capture(alert) -> RunResult:
        """In place of the loop: Klaxon's own tools, called once each, while the fault is in place."""
        began = time.monotonic()
        captured["alert"] = asdict(alert)
        captured["money_check_called_at"] = iso(time.time())
        captured["money_check"] = tools["money_check"].call({})
        captured["money_check_seconds"] = round(time.monotonic() - began, 1)
        captured["recent_deploys_called_at"] = iso(time.time())
        captured["recent_deploys"] = tools["recent_deploys"].call({})
        logs_at = time.time()
        captured["logs"] = read_logs(sources, logs_from, logs_at)
        captured["logs_read_at"] = iso(logs_at)
        return RunResult(
            "captured",
            "Klaxon's tools were called by scripts/capture_incident.py in place of a model",
            None,
            0,
            3,
            round(time.monotonic() - began, 1),
            Path("no-trace"),
        )

    started = datetime.now(timezone.utc)
    # Klaxon's runner, unchanged: it clears the fault and stops the traffic, whatever happens.
    fault_runs.run(fault, record, stack, settings, capture, warmup_seconds=args.warmup_seconds)
    if "logs" not in captured:
        print(f"Nothing captured: {record.stopped}")
        if args.record_on_failure:
            args.record_on_failure.write_text(json.dumps(asdict(record), indent=2, default=str) + "\n")
            print(f"The runner's record: {args.record_on_failure}")
        return 1
    record.stopped = "captured at the alert: Klaxon's tools were called by this script in place of a model"

    args.out.mkdir(parents=True)
    by_service = Counter(line["service"] for line in captured["logs"])
    by_level = Counter(f"{line['service']} {line['level']}" for line in captured["logs"])
    incident = {
        "name": fault.name,
        "title": fault.title,
        "label": LABEL,
        "fault": fault.describe(),
        "alert": captured["alert"],
        "captured": {
            "klaxon_commit": commit,
            "klaxon_code_clean": git("status", "--porcelain", "--", "src", "compose.yaml", "docker") == "",
            "script": "scripts/capture_incident.py",
            "command": shlex.join(["python", "scripts/capture_incident.py", *map(shown, argv or sys.argv[1:])]),
            "started_at_utc": started.isoformat(timespec="seconds"),
            "traffic": {"rate_per_second": args.rate, "seed": args.seed},
            "warmup_seconds": args.warmup_seconds,
            "klaxon_default_warmup_seconds": fault_runs.WARMUP_SECONDS,
            "note": args.note,
        },
        "money_check_called_at": captured["money_check_called_at"],
        "money_check_seconds": captured["money_check_seconds"],
        "recent_deploys_called_at": captured["recent_deploys_called_at"],
        "logs_read_from": iso(logs_from),
        "logs_read_at": captured["logs_read_at"],
        "log_lines": len(captured["logs"]),
        "log_lines_by_service": dict(sorted(by_service.items())),
        "log_lines_by_service_and_level": dict(sorted(by_level.items())),
    }
    (args.out / "incident.json").write_text(json.dumps(incident, indent=2, ensure_ascii=False) + "\n")
    (args.out / "money_check.json").write_text(json.dumps(captured["money_check"], indent=2, ensure_ascii=False) + "\n")
    (args.out / "recent_deploys.json").write_text(
        json.dumps(captured["recent_deploys"], indent=2, ensure_ascii=False) + "\n"
    )
    with open(args.out / "logs.jsonl", "w", encoding="utf-8") as out:
        for line in captured["logs"]:
            out.write(json.dumps(line, ensure_ascii=False, separators=(",", ":")) + "\n")
    harness = {
        "label": (
            "Klaxon's fault-run record (klaxon.fault_runs.Record) for this capture: every evaluation of the alert"
            " rules, the timings in seconds since the traffic started, the injection and its clearing, the traffic."
            " Its loop entry is this script's capture, not a model's run."
        ),
        "record": asdict(record),
    }
    (args.out / "harness.json").write_text(json.dumps(harness, indent=2, default=str) + "\n")
    print(f"Alert: {captured['alert']['name']}: {captured['alert']['summary']}")
    print(
        f"Money check: {captured['money_check'].get('outcome')}; log lines: {len(captured['logs'])} {dict(by_service)}"
    )
    print(f"Cleared: {record.cleared}")
    print(f"Saved in {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
