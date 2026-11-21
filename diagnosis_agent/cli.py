"""The command line: python -m diagnosis_agent <command>.

    cases                         the three cases: each alert, right answer and remedy; calls nothing
    prompt CASE                   the exact system prompt and first turn a run gets; calls no model
    tool CASE NAME [JSON]         call one of the three tools on a case's data, and print its result
    run CASE --model scripted     one offline run with the scripted mock model (no model is called)
    run CASE --model MODEL_ID     one live run: Claude through Claude Code on the seat (claude-opus-5-5)
    replay FOLDER                 replay a recorded live run offline, and check that it reproduces exactly
    show FOLDER                   a run step by step, from its files: each reason, decision, call and result
    check FOLDER [FOLDER ...]     check runs against their evidence (version 2 unless --version 1); --write saves
                                  the result beside each run (check-v2.json, or check.json for version 1)
    report [FOLDER ...]           one line per run, and counts per case (default: every run in runs/)

New for this mini-project, except where it reuses Klaxon's code (the loop, the adapters, record and replay).
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
import tempfile
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from diagnosis_agent import KLAXON_COMMIT, REPO, cases, check, loop, scripted
from diagnosis_agent.model import ModelError
from diagnosis_agent.text import clean_text
from diagnosis_agent.tools import data_sha256

RUNS, SCRIPTED_RUNS = REPO / "runs", REPO / "runs-scripted"


def _new_folder(parent: Path, stem: str) -> Path:
    folder, n = parent / stem, 2
    while folder.exists():
        folder, n = parent / f"{stem}-{n}", n + 1
    folder.mkdir(parents=True)
    return folder


def _print_diagnosis(found: dict) -> None:
    """A diagnosis, as the loop accepted it: a dict, as Diagnosis.as_dict() gives it and the trace keeps it."""
    print(f"Hypothesis: {clean_text(found['hypothesis'])}")
    print(f"Confidence: {found['confidence']:g}")
    fix = found["proposed_fix"]
    print(f"Proposed fix (not applied): {fix['kind']} {clean_text(fix['target'], 200)}: {clean_text(fix['summary'])}")
    print(f"Blast radius: {clean_text(found['blast_radius'])}")
    print(f"Rollback plan: {clean_text(found['rollback_plan'])}")
    print("Evidence:")
    for item in found["evidence"]:
        print(f"  {item['call_id']} {item['tool']}: {clean_text(item['shows'])}")


def _print_result(result: loop.RunResult, folder: Path | None = None) -> None:
    print(f"Stopped: {result.reason} ({clean_text(result.detail, 300)})")
    print(f"Steps: {result.steps}, tool calls: {result.tool_calls}, {result.seconds:.1f} s")
    if result.diagnosis is not None:
        _print_diagnosis(result.diagnosis.as_dict())
    if folder is not None:
        print(f"Saved in {clean_text(str(folder))}")


WHEN = {1: "committed before the live runs", 2: "written after them"}


def _print_check(result: dict) -> None:
    verdict = "PASSED every check" if result["passed"] else "FAILED a check"
    version = result.get("version", 1)
    print(f"\nChecked against the evidence, by code: version {version} ({WHEN[version]}): {verdict}")
    for item in result["checks"]:
        print(f"  [{'pass' if item['passed'] else 'FAIL'}] {item['check']}: {clean_text(item['detail'], 300)}")


def _result_line(call: dict) -> str:
    """What one tool call returned, in one line, from the run's own record of the result."""
    text = call["text"]
    if call["is_error"]:
        return ("error, simulated" if call.get("simulated_by") else "error") + f": {clean_text(text, 200)}"
    try:
        body = json.loads(text)
    except ValueError:
        body = None
    if not isinstance(body, dict):
        return clean_text(text, 200)
    if call["tool"] == "money_check":
        invariants = [item for item in body.get("invariants") or [] if isinstance(item, dict)]
        failed = [item for item in invariants if not item.get("passed")]
        if body.get("outcome") == "could not run":
            return clean_text(f"could not run: {body.get('reason', '')}", 300)
        if failed:
            counts = ", ".join(f"{item.get('name')} ({len(item.get('evidence') or [])} findings)" for item in failed)
            return clean_text(f"{body.get('outcome')}: {len(failed)} of {len(invariants)} invariants: {counts}", 300)
        return clean_text(f"{body.get('outcome')}: all {len(invariants)} invariants held", 300)
    if call["tool"] == "search_logs":
        lines = [line for line in body.get("lines") or [] if isinstance(line, dict)]
        said = f"{body.get('matched')} line(s) matched"
        if lines:
            newest, fields = lines[0], lines[0].get("fields") or {}
            more = [
                f"{key} {fields[key]}" for key in ("event_id", "source", "error", "reason", "mode") if key in fields
            ]
            said += (
                f"; newest: {str(newest.get('time', ''))[11:23]} {newest.get('service')} {newest.get('level')}"
                f" {newest.get('message')}" + (f" ({', '.join(more)})" if more else "")
            )
        return clean_text(said, 300)
    if call["tool"] == "recent_deploys":
        deploys = [item for item in body.get("deploys") or [] if isinstance(item, dict)]
        if not deploys:
            return "no deploys listed"
        newest = deploys[0]
        settings = ", ".join(f"{key}={value}" for key, value in (newest.get("settings") or {}).items())
        when = str(newest.get("deployed_at", ""))[11:19]
        return clean_text(
            f"{len(deploys)} deploys; newest: {newest.get('service')} at {when}"
            + (f", {settings}" if settings else ""),
            300,
        )
    return clean_text(text, 200)


def _print_steps(run: dict) -> None:
    """Each step of a run, from its trace: the model's reason, the loop's decision, the call and what came back."""
    for step in run["trace"]:
        if step.get("kind") != "step":
            continue
        proposed, ran = step.get("proposed") or {}, step.get("ran")
        if ran:
            arguments = json.dumps(ran.get("arguments") or {}, ensure_ascii=False)
            print(f"\nStep {step['step']}: {ran['tool']} {clean_text(arguments, 300)}")
        else:
            print(f"\nStep {step['step']}: {proposed.get('action') or 'no usable reply'}")
        if step.get("error"):
            print(f"  model call failed: {clean_text(step['error'], 300)}")
        if proposed.get("why"):
            print(f"  why:  {clean_text(str(proposed['why']), 400)}")
        if step.get("why"):
            decision = "allowed" if step.get("allowed") else "refused" if step.get("allowed") is False else "loop"
            print(f"  {decision}: {clean_text(step['why'], 300)}")
        if ran:
            print(f"  back: {ran['call_id']}, {_result_line(run['calls'][ran['call_id']])}")


def _cases(_: argparse.Namespace) -> int:
    for case in cases.CASES.values():
        alert = case.alert()
        print(f"{case.name} ({case.kind}): {case.title}")
        print(f"  alert: {alert.name}: {alert.summary}")
        if case.withheld:
            print(f'  withheld: {", ".join(case.withheld)} answers "{next(iter(case.withheld.values()))}" (simulated)')
        print(f"  right answer: {case.right_answer}")
        print(f"  remedy: {case.remedy}\n")
    return 0


def _prompt(args: argparse.Namespace) -> int:
    case = cases.get(args.case)
    tools = case.tools()
    specs = loop.allowed_tools(tools.open())
    system = loop.system_prompt(specs)
    print(f"System prompt ({len(system):,} characters):\n\n{system}\n")
    print(f"First turn:\n\n{loop.first_turn(case.alert(), cases.BUDGETS)}")
    return 0


def _tool(args: argparse.Namespace) -> int:
    case = cases.get(args.case)
    try:
        arguments = json.loads(args.arguments)
    except ValueError as exc:
        print(f"TOOL ERROR: the arguments are not JSON: {clean_text(str(exc), 200)}")
        return 1
    tools = case.tools()
    tools.open()
    result = tools.call(args.name, arguments, loop.TOOL_TIMEOUT)
    try:
        shown = json.dumps(json.loads(result.text), indent=2, ensure_ascii=False)
    except ValueError:
        shown = result.text
    print(
        ("TOOL ERROR" + (f" (simulated: {result.simulated_by})" if result.simulated_by else "") + ": ")
        * result.is_error
        + shown
    )
    return 1 if result.is_error else 0


def _header(case: cases.Case, model: str, argv: list[str]) -> dict:
    from diagnosis_agent.claude_code import _provenance

    return {
        "what": (
            f"One run of Klaxon's diagnosis agent on the case {case.name}: "
            + (
                "the scripted mock model (no model was called)."
                if model == "scripted"
                else f"{model} through Claude Code on Aravind's seat (live)."
            )
            + " Its tools read incident data captured from Klaxon's live stack."
        ),
        "case": case.name,
        "model": model,
        "budgets": asdict(cases.BUDGETS),
        "rerun_command": clean_text(shlex.join(["python", "-m", "diagnosis_agent", *argv]), 2000),
        "incident": case.incident,
        "incident_data_sha256": data_sha256(case.folder()),
        "klaxon_commit_copied_from": KLAXON_COMMIT,
        **_provenance(),
    }


def _run(args: argparse.Namespace, argv: list[str]) -> int:
    case = cases.get(args.case)
    alert = case.alert()
    live = args.model != "scripted"
    binary = None
    if live:
        from diagnosis_agent.claude_code import REQUESTED, find_claude

        try:  # before any folder is made: a run that cannot start leaves nothing behind
            if not REQUESTED.fullmatch(args.model):
                raise ValueError("--model must be scripted, or a full model id such as claude-opus-5-5")
            binary = find_claude()
        except (ModelError, ValueError) as exc:
            print(exc, file=sys.stderr)
            return 2
    parent = args.out or (RUNS if live else SCRIPTED_RUNS)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    folder = _new_folder(parent, f"{stamp}_{case.name}_{args.model}")
    header = _header(case, args.model, argv)
    run_file = folder / check.RUN_FILE
    run_file.write_text(json.dumps({"case": case.name, "model": args.model, "header": header}, indent=2) + "\n")
    try:
        if live:
            result = loop.investigate_live(alert, cases.BUDGETS, folder, header, args.model, binary, case.tools())
        else:
            model = scripted.model_for(case)
            result = loop.investigate_offline(alert, cases.BUDGETS, folder, header, model, case.tools())
    except KeyboardInterrupt:
        print(f"Interrupted; the trace says so: {clean_text(str(folder))}", file=sys.stderr)
        return 130
    summary = {
        "reason": result.reason,
        "detail": result.detail,
        "steps": result.steps,
        "tool_calls": result.tool_calls,
        "seconds": result.seconds,
        "diagnosis": None if result.diagnosis is None else result.diagnosis.as_dict(),
    }
    run_file.write_text(
        json.dumps({"case": case.name, "model": args.model, "header": header, "result": summary}, indent=2) + "\n"
    )
    _print_steps(check.read_run(folder))
    print()
    _print_result(result, folder)
    check.write_check(folder, version=1)  # the version committed before the live runs, kept for every run
    checked = check.write_check(folder, version=2)
    _print_check(checked)
    return 0 if checked["passed"] else 1


def _replay(args: argparse.Namespace) -> int:
    try:
        with tempfile.TemporaryDirectory(prefix="diagnosis-replay-") as scratch:
            result, whole, differences = loop.replay(args.folder, Path(scratch) / loop.TRACE)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(
            f"Cannot replay {clean_text(str(args.folder))}: {type(exc).__name__}: {clean_text(str(exc), 300)}",
            file=sys.stderr,
        )
        return 2
    _print_result(result)
    if differences:
        print("Not reproduced:\n" + "\n".join(f"  {difference}" for difference in differences))
        return 1
    if not whole:
        print(
            "Reproduced up to the step the recorded run stopped in (it stopped on its time budget, an interrupt or a"
            " crash)."
        )
        return 0
    print("Reproduced: the same prompts, replies, decisions, tool results and stop as the recorded run.")
    return 0


def _show(args: argparse.Namespace) -> int:
    if not check.is_run(args.folder):
        print(
            f"Not a run folder (no trace, tool results and run.json): {clean_text(str(args.folder))}", file=sys.stderr
        )
        return 2
    run = check.read_run(args.folder)
    start, stop = run["trace"][0], run["trace"][-1]
    alert = start.get("extra", {}).get("alert", {})
    print(f"{args.folder.name}: case {run['case'].name}, model {start.get('model')} ({start.get('source')})")
    print(f"Alert: {alert.get('name')}, fired at {alert.get('fired_at')}: {clean_text(str(alert.get('summary', '')))}")
    _print_steps(run)
    totals = stop.get("totals", {})
    print(f"\nStopped: {stop.get('reason')} ({clean_text(stop.get('detail', ''), 300)})")
    print(f"Steps: {totals.get('steps')}, tool calls: {totals.get('tool_calls')}, {totals.get('seconds', 0):.1f} s")
    if stop.get("diagnosis"):
        _print_diagnosis(stop["diagnosis"])
    for version in check.CHECK_FILES:
        _print_check(check.check_run(args.folder, version=version))
    return 0


def _check(args: argparse.Namespace) -> int:
    failed = 0
    for folder in args.folders:
        if not check.is_run(folder):
            print(f"Not a run folder (no trace, tool results and run.json): {clean_text(str(folder))}", file=sys.stderr)
            return 2
        if args.write:
            result = check.write_check(folder, version=args.version)
        else:
            result = check.check_run(folder, version=args.version)
        print(f"{folder.name}:")
        _print_check(result)
        failed += not result["passed"]
    return 1 if failed else 0


def run_folders(roots: list[Path]) -> list[Path]:
    found = []
    for root in roots:
        if check.is_run(root):
            found.append(root)
        elif root.is_dir():
            found += sorted(path for path in root.iterdir() if path.is_dir() and check.is_run(path))
    return found


# Stops that say nothing of the agent's judgment, as Klaxon's evaluation counts them: such a run is "not run".
NOT_RUN = ("model_error", "tools_failed", "crashed", "interrupted")


def usage(folder: Path) -> dict:
    """A run's model calls, time and cost, from its trace: replies, failed calls, and Claude Code's estimate at API
    prices (a seat is not billed per call)."""
    steps = [record for record in check.read_records(folder / loop.TRACE) if record.get("kind") == "step"]
    replies = [step["reply"] for step in steps if step.get("reply")]
    return {
        "replies": len(replies),
        "failed_calls": sum(1 for step in steps if step.get("error")),
        "cost_usd": round(sum(reply.get("cost_usd") or 0 for reply in replies), 4),
    }


def _failed(result: dict) -> str:
    failed = [item["check"] for item in result["checks"] if not item["passed"]]
    return "passed" if result["passed"] else "failed: " + ", ".join(failed)


def _report(args: argparse.Namespace) -> int:
    folders = run_folders(args.folders or [RUNS])
    if not folders:
        print("No runs found.")
        return 1
    rows = []
    for folder in folders:
        one, two = check.check_run(folder, version=1), check.check_run(folder, version=2)
        rows.append((folder, one, two, usage(folder)))
    if args.markdown:
        print("| Run | Case | Stop | Steps | Tool calls | Path | Rule | Checks v1 | Checks v2 | Confidence | Seconds |")
        print("|---|---|---|---|---|---|---|---|---|---|---|")
        for folder, one, two, _ in rows:
            verdict = one.get("cause", {}).get("verdict", "-")
            stop = "not run: " + one["stop_reason"] if one["stop_reason"] in NOT_RUN else one["stop_reason"]
            print(
                f"| {folder.name[:18]} | {one['case']} | {stop} | {one['steps']} | {one['tool_calls']} |"
                f" {' > '.join(one['path']) or '-'} | {verdict} | {_failed(one)} | {_failed(two)} |"
                f" {one['confidence'] if one['confidence'] is not None else '-'} | {one['seconds']:.0f} |"
            )
    else:
        for folder, one, two, _ in rows:
            print(
                f"{folder.name}: {one['stop_reason']}, {one['steps']} steps, path {' > '.join(one['path']) or '-'},"
                f" confidence {one['confidence']}; v1 {_failed(one)}; v2 {_failed(two)}"
            )
    print()
    by_case: dict[tuple[str, str], list[tuple]] = defaultdict(list)
    for row in rows:
        by_case[(row[1]["case"], row[1]["model"])].append(row)
    for (name, model), group in sorted(by_case.items()):
        ran = [row for row in group if row[1]["stop_reason"] not in NOT_RUN]
        named = sum(row[1].get("cause", {}).get("verdict") == "names the cause" for row in ran)
        one = sum(row[1]["passed"] for row in ran)
        two = sum(row[2]["passed"] for row in ran)
        not_run = len(group) - len(ran)
        print(
            f"{name} on {model}: {len(ran)} runs" + (f" ({not_run} more not run)" if not_run else "") + ";"
            f" the rule says {named} of {len(ran)} name the cause; every check passed in {one} of {len(ran)} by"
            f" version 1, {two} of {len(ran)} by version 2"
        )
    calls = sum(row[3]["replies"] + row[3]["failed_calls"] for row in rows)
    cost = sum(row[3]["cost_usd"] for row in rows)
    print(f"In all: {len(rows)} runs, {calls} model calls, ${cost:.2f} at API prices (Claude Code's estimate)")
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(prog="python -m diagnosis_agent", description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("cases", help="the three cases; calls nothing")
    prompt = commands.add_parser("prompt", help="the system prompt and first turn; calls no model")
    prompt.add_argument("case", choices=list(cases.CASES))
    tool = commands.add_parser("tool", help="call one tool on a case's data")
    tool.add_argument("case", choices=list(cases.CASES))
    tool.add_argument("name")
    tool.add_argument("arguments", nargs="?", default="{}")
    run = commands.add_parser("run", help="one run: --model scripted (offline) or a full Claude model id (live)")
    run.add_argument("case", choices=list(cases.CASES))
    run.add_argument("--model", required=True, help="scripted, or a full model id such as claude-opus-5-5")
    run.add_argument(
        "--out", type=Path, help=f"where the run's folder goes (default {RUNS.name}/ or {SCRIPTED_RUNS.name}/)"
    )
    again = commands.add_parser("replay", help="replay a recorded live run offline")
    again.add_argument("folder", type=Path)
    show = commands.add_parser("show", help="a run step by step, from its files; calls nothing")
    show.add_argument("folder", type=Path)
    checks = commands.add_parser("check", help="check runs against their evidence")
    checks.add_argument("folders", type=Path, nargs="+")
    checks.add_argument("--write", action="store_true", help="save the result beside each run, as its check file")
    checks.add_argument("--version", type=int, choices=(1, 2), default=2, help="version of the checks (default 2)")
    report = commands.add_parser("report", help="one line per run, and counts per case")
    report.add_argument("folders", type=Path, nargs="*")
    report.add_argument("--markdown", action="store_true", help="a table, for the docs")
    args = parser.parse_args(argv)
    if args.command == "run":
        return _run(args, argv)
    commands_by_name = {"cases": _cases, "prompt": _prompt, "tool": _tool, "replay": _replay, "show": _show}
    return (commands_by_name | {"check": _check, "report": _report})[args.command](args)
