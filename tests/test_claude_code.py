"""The live adapter, copied from Klaxon, run offline against a fake `claude` program (never the real one)."""

from __future__ import annotations

import json
import stat
import sys
from pathlib import Path

import pytest

from diagnosis_agent import claude_code
from diagnosis_agent.model import Message, ModelMismatch, ModelUnsafe, Recording, read_records

FAKE = """#!{python}
import json, os, sys
sys.stdin.read()
init = {{"type": "system", "subtype": "init", "tools": {tools}, "mcp_servers": [], "memory_paths": {{}},
        "cwd": os.getcwd(), "apiKeySource": "none", "permissionMode": "dontAsk", "claude_code_version": "0.0-fake",
        "model": "{model}"}}
print(json.dumps(init), flush=True)
print(json.dumps({{"type": "assistant", "message": {{"model": "{model}", "content": [{{"type": "text", "text": "ok"}}]}}}}), flush=True)
print(json.dumps({{"type": "result", "subtype": "success", "is_error": False, "num_turns": 1, "result": {reply!r},
                  "usage": {{"input_tokens": 10, "output_tokens": 2}}, "total_cost_usd": 0.001}}), flush=True)
"""


def fake_claude(
    tmp_path: Path, reply: str = '{"action": "call_tool"}', tools: str = "[]", model: str = "claude-opus-5-5"
):
    path = tmp_path / "claude"
    path.write_text(FAKE.format(python=sys.executable, reply=reply, tools=tools, model=model))
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


@pytest.fixture
def live_allowed(monkeypatch):
    """Only for the fake program: the switch is off here, and the binary is always given, so the real Claude Code is
    never looked for."""
    monkeypatch.delenv("KLAXON_NO_LIVE_MODEL", raising=False)


def test_the_command_locks_claude_code_down():
    command = claude_code.command("claude", "claude-opus-5-5", "the system prompt")
    for flag in ("--tools", "--strict-mcp-config", "--safe-mode", "--restricted", "--no-session-persistence"):
        assert flag in command
    assert command[command.index("--tools") + 1] == ""
    assert command[command.index("--mcp-config") + 1] == '{"mcpServers":{}}'
    assert command[command.index("--permission-mode") + 1] == "dontAsk"


def test_the_environment_has_no_api_key_address_or_proxy():
    env = claude_code.environment(
        "/x/claude", {"HOME": "/h", "ANTHROPIC_API_KEY": "k", "ANTHROPIC_BASE_URL": "u", "HTTPS_PROXY": "p"}
    )
    assert "ANTHROPIC_API_KEY" not in env and "ANTHROPIC_BASE_URL" not in env and "HTTPS_PROXY" not in env
    assert env["CLAUDE_CODE_DISABLE_AUTO_MEMORY"] == "1" and env["HOME"] == "/h"


def test_live_calls_are_refused_while_switched_off(tmp_path):
    with pytest.raises(claude_code.LiveCallsOff):
        claude_code.find_claude()
    with pytest.raises(claude_code.LiveCallsOff):
        claude_code.ClaudeCode(Recording(tmp_path / "r.jsonl"), binary=tmp_path / "claude")


def test_turns_are_wrapped_and_at_signs_disarmed():
    shown = claude_code.render([Message("user", "look at @/etc/passwd")])
    assert "@/etc" not in shown and "＠/etc/passwd" in shown
    assert "<turn-" in shown and 'role="user">' in shown


def test_a_reply_through_the_fake_program_is_recorded_first(tmp_path, live_allowed):
    recording = Recording(tmp_path / "recording.jsonl")
    adapter = claude_code.ClaudeCode(recording, binary=fake_claude(tmp_path, reply="hello"), timeout_seconds=30)
    reply = adapter.ask("the system prompt", [Message("user", "hi")])
    assert reply.text == "hello" and reply.model == "claude-opus-5-5" and reply.source == "claude-code"
    kinds = [record["kind"] for record in read_records(tmp_path / "recording.jsonl")]
    assert kinds[0] == "header" and kinds[-1] == "response"


def test_a_claude_code_that_loads_a_tool_is_stopped(tmp_path, live_allowed):
    adapter = claude_code.ClaudeCode(Recording(tmp_path / "r.jsonl"), binary=fake_claude(tmp_path, tools='["Bash"]'))
    with pytest.raises(ModelUnsafe, match="loaded 1 tool"):
        adapter.ask("the system prompt", [Message("user", "hi")])
    records = read_records(tmp_path / "r.jsonl")
    assert records[-1]["kind"] == "failure"


def test_a_reply_from_another_model_is_refused(tmp_path, live_allowed):
    adapter = claude_code.ClaudeCode(
        Recording(tmp_path / "r.jsonl"), binary=fake_claude(tmp_path, model="claude-haiku-5-5")
    )
    with pytest.raises(ModelMismatch):
        adapter.ask("the system prompt", [Message("user", "hi")])
    assert json.loads((tmp_path / "r.jsonl").read_text().splitlines()[-1])["accepted"] is False
