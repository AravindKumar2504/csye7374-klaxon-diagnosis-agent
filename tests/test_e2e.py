"""End to end: the command line as a person runs it, each command in its own process, on the repo's real data.

Every command in README.md that needs no model is run here, exactly as written there, and checked for its exit code and
its words. A live run is refused while live calls are switched off (conftest.py switches them off for every test), and
leaves nothing behind.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from diagnosis_agent import REPO, cases, check, cli

LIVE = [folder for folder in cli.run_folders([cli.RUNS]) if check.has_recording(folder)]


def cli_run(*args: str, cwd: Path = REPO) -> subprocess.CompletedProcess:
    env = os.environ | {"KLAXON_NO_LIVE_MODEL": "1"}
    return subprocess.run(
        [sys.executable, "-m", "diagnosis_agent", *args], cwd=cwd, env=env, capture_output=True, text=True, timeout=120
    )


def test_cases_lists_the_three_cases_and_their_alerts():
    done = cli_run("cases")
    assert done.returncode == 0, done.stderr
    for name in cases.CASES:
        assert f"{name} (" in done.stdout
    assert "alert: MoneyCheckFailed:" in done.stdout and "alert: BooksBehindProvider:" in done.stdout
    assert "withheld: recent_deploys" in done.stdout


def test_prompt_prints_the_system_prompt_with_the_three_tools():
    done = cli_run("prompt", "poison-message")
    assert done.returncode == 0, done.stderr
    for tool in ("money_check", "search_logs", "recent_deploys"):
        assert f'"name": "{tool}"' in done.stdout
    assert "First turn:" in done.stdout and "BooksBehindProvider" in done.stdout


def test_tool_answers_on_the_real_data_and_refuses_bad_arguments():
    done = cli_run(
        "tool", "poison-message", "search_logs", '{"service": "ledger-consumer", "level": "ERROR", "limit": 2}'
    )
    assert done.returncode == 0, done.stderr
    body = json.loads(done.stdout)
    assert body["matched"] >= 2 and len(body["lines"]) == 2
    assert all(line["service"] == "ledger-consumer" and line["level"] == "ERROR" for line in body["lines"])
    bad = cli_run("tool", "poison-message", "search_logs", '{"service": "ledger"}')
    assert bad.returncode == 1 and "service must be one of" in bad.stdout


def test_the_withheld_deploy_list_answers_the_simulated_outage():
    done = cli_run("tool", "missing-deploys", "recent_deploys")
    assert done.returncode == 1
    assert "simulated" in done.stdout and "docker did not answer within 15 s" in done.stdout


@pytest.mark.parametrize("name", list(cases.CASES))
def test_a_scripted_run_end_to_end_passes_every_check(tmp_path, name):
    done = cli_run("run", name, "--model", "scripted", "--out", str(tmp_path))
    assert done.returncode == 0, done.stdout + done.stderr
    assert "Stopped: diagnosed" in done.stdout and "PASSED every check" in done.stdout
    (folder,) = list(tmp_path.iterdir())
    for name_ in ("trace.jsonl", "tool-results.jsonl", "run.json", "check.json", "check-v2.json"):
        assert (folder / name_).is_file(), name_
    assert not (folder / "model-recording.jsonl").exists()  # a scripted run records no model: none was called


def test_a_live_run_is_refused_while_live_calls_are_off(tmp_path):
    done = cli_run("run", "poison-message", "--model", "claude-opus-5-5", "--out", str(tmp_path))
    assert done.returncode == 2
    assert "switched off" in done.stderr
    assert list(tmp_path.iterdir()) == []  # refused before any folder was made


@pytest.mark.parametrize("folder", LIVE, ids=[folder.name for folder in LIVE])
def test_replay_from_the_command_line_reproduces(folder):
    done = cli_run("replay", str(folder.relative_to(REPO)))
    assert done.returncode == 0, done.stdout + done.stderr
    assert "Reproduced" in done.stdout


@pytest.mark.parametrize("folder", LIVE, ids=[folder.name for folder in LIVE])
def test_check_from_the_command_line_says_what_the_saved_check_says(folder):
    for version, name in check.CHECK_FILES.items():
        saved = json.loads((folder / name).read_text())
        done = cli_run("check", str(folder.relative_to(REPO)), "--version", str(version))
        assert done.returncode == (0 if saved["passed"] else 1)
        assert ("PASSED every check" if saved["passed"] else "FAILED a check") in done.stdout


def test_report_counts_the_runs_as_the_readme_says():
    done = cli_run("report")
    assert done.returncode == 0, done.stderr
    assert (
        "duplicate-delivery on claude-opus-5-5: 3 runs; the rule says 3 of 3 name the cause; every check passed in 0 of 3 by version 1, 3 of 3 by version 2"
        in done.stdout
    )
    assert (
        "poison-message on claude-opus-5-5: 3 runs (1 more not run); the rule says 3 of 3 name the cause; every check passed in 3 of 3 by version 1, 3 of 3 by version 2"
        in done.stdout
    )
    assert (
        "missing-deploys on claude-opus-5-5: 3 runs; the rule says 3 of 3 name the cause; every check passed in 0 of 3 by version 1, 2 of 3 by version 2"
        in done.stdout
    )
    assert "In all: 10 runs, 73 model calls, $3.58" in done.stdout
    table = cli_run("report", "--markdown")
    assert table.returncode == 0 and table.stdout.startswith("| Run | Case |")


def test_the_project_document_builds_from_the_record_with_no_placeholder(tmp_path):
    env = os.environ | {"KLAXON_NO_LIVE_MODEL": "1"}
    out = tmp_path / "report.html"
    done = subprocess.run(
        [sys.executable, "scripts/build_report.py", str(out)],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert done.returncode == 0, done.stderr
    page = out.read_text()
    assert "Aravind writes" not in page and "—" not in page
    assert "AI was used in this project" in page
