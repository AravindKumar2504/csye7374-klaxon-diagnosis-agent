"""Every recorded live run in runs/ replays offline, exactly, and its saved check is what the checks give today."""

from __future__ import annotations

import json

import pytest

from diagnosis_agent import check, cli, loop

LIVE = [folder for folder in cli.run_folders([cli.RUNS]) if check.has_recording(folder)]


def test_there_are_live_runs_to_replay():
    if not cli.RUNS.exists():
        pytest.skip("no runs/ folder yet: the live runs have not been made")
    assert LIVE, "runs/ holds no live run"


@pytest.mark.parametrize("folder", LIVE, ids=[folder.name for folder in LIVE])
def test_a_live_run_replays_exactly(tmp_path, folder):
    result, whole, differences = loop.replay(folder, tmp_path / loop.TRACE)
    assert differences == []
    assert whole


@pytest.mark.parametrize("folder", LIVE, ids=[folder.name for folder in LIVE])
def test_its_saved_check_is_the_check_today(folder):
    saved = json.loads((folder / check.CHECK_FILE).read_text())
    assert check.check_run(folder) == saved


@pytest.mark.parametrize("folder", LIVE, ids=[folder.name for folder in LIVE])
def test_it_read_the_incident_data_this_repo_holds(folder):
    start = json.loads((folder / loop.TRACE).read_text().splitlines()[0])
    recorded = start["extra"]["tool_server"]["data_sha256"]
    about = json.loads((folder / check.RUN_FILE).read_text())
    assert recorded == about["header"]["incident_data_sha256"]
    from diagnosis_agent import cases
    from diagnosis_agent.tools import data_sha256

    assert recorded == data_sha256(cases.get(about["case"]).folder())
