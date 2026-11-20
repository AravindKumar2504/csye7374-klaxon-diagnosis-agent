"""The command line: python -m diagnosis_agent <command>.

    cases                         the three cases: each alert, right answer and remedy; calls nothing
    prompt CASE                   the exact system prompt and first turn a run gets; calls no model
    tool CASE NAME [JSON]         call one of the three tools on a case's data, and print its result
    run CASE --model scripted     one offline run with the scripted mock model (no model is called)
    run CASE --model MODEL_ID     one live run: Claude through Claude Code on the seat (claude-opus-5-5)
    replay FOLDER                 replay a recorded live run offline, and check that it reproduces exactly
    check FOLDER [FOLDER ...]     check runs against their evidence; --write saves check.json beside each
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


def _print_result(result: loop.RunResult, folder: Path | None = None) -> None:
    print(f"Stopped: {result.reason} ({clean_text(result.detail, 300)})")
    print(f"Steps: {result.steps}, tool calls: {result.tool_calls}, {result.seconds:.1f} s")
    if result.diagnosis is not None:
        found = result.diagnosis
        print(f"Hypothesis: {clean_text(found.hypothesis)}")
        print(f"Confidence: {found.confidence:g}")
        fix = found.proposed_fix
        print(f"Proposed fix (not applied): {fix.kind} {clean_text(fix.target, 200)}: {clean_text(fix.summary)}")
        for item in found.evidence:
            print(f"  {item.call_id} {item.tool}: {clean_text(item.shows)}")
    if folder is not None:
        print(f"Saved in {clean_text(str(folder))}")


def _print_check(result: dict) -> None:
    verdict = "PASSED every check" if result["passed"] else "FAILED a check"
    print(f"\nChecked against the evidence ({result['label'].split(':')[0]}): {verdict}")
    for item in result["checks"]:
        print(f"  [{'pass' if item['passed'] else 'FAIL'}] {item['check']}: {clean_text(item['detail'], 300)}")


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
    tools = case.tools()
    tools.open()
    result = tools.call(args.name, json.loads(args.arguments), loop.TOOL_TIMEOUT)
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
    _print_result(result, folder)
    checked = check.write_check(folder)
    _print_check(checked)
    return 0 if checked["passed"] else 1


def _replay(args: argparse.Namespace) -> int:
    try:
        with tempfile.TemporaryDirectory(prefix="diagnosis-replay-") as scratch:
            result, whole, differences = loop.replay(args.folder, Path(scratch) / loop.TRACE)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"Cannot replay {clean_text(str(args.folder))}: {type(exc).__name__}: {clean_text(str(exc), 300)}")
        return 2
    _print_result(result)
    if differences:
        print("Not reproduced:\n" + "\n".join(f"  {difference}" for difference in differences))
        return 1
    if not whole:
        print("Reproduced up to the step the recorded run stopped in (it stopped on its time budget or an interrupt).")
        return 0
    print("Reproduced: the same prompts, replies, decisions, tool results and stop as the recorded run.")
    return 0


def _check(args: argparse.Namespace) -> int:
    failed = 0
    for folder in args.folders:
        result = check.write_check(folder) if args.write else check.check_run(folder)
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


def _report(args: argparse.Namespace) -> int:
    folders = run_folders(args.folders or [RUNS])
    if not folders:
        print("No runs found.")
        return 1
    by_case: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for folder in folders:
        result = check.check_run(folder)
        by_case[(result["case"], result["model"])].append(result)
        failed = [item["check"] for item in result["checks"] if not item["passed"]]
        print(
            f"{folder.name}: {result['stop_reason']}, {result['steps']} steps, path {' > '.join(result['path']) or '-'},"
            f" confidence {result['confidence']}, {'passed' if result['passed'] else 'FAILED ' + ', '.join(failed)}"
        )
    print()
    for (name, model), results in sorted(by_case.items()):
        passed = sum(result["passed"] for result in results)
        named = sum(result.get("cause", {}).get("verdict") == "names the cause" for result in results)
        print(
            f"{name} on {model}: {passed} of {len(results)} passed every check; the rule says {named} of {len(results)} name the cause"
        )
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
    checks = commands.add_parser("check", help="check runs against their evidence")
    checks.add_argument("folders", type=Path, nargs="+")
    checks.add_argument("--write", action="store_true", help="save check.json beside each run")
    report = commands.add_parser("report", help="one line per run, and counts per case")
    report.add_argument("folders", type=Path, nargs="*")
    args = parser.parse_args(argv)
    if args.command == "run":
        return _run(args, argv)
    return {"cases": _cases, "prompt": _prompt, "tool": _tool, "replay": _replay, "check": _check, "report": _report}[
        args.command
    ](args)
