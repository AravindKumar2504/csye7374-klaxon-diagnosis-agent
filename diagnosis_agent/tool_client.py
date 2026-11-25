# Copied from Klaxon (github.com/AravindKumar2504/klaxon) at commit fef450b (fef450b0aae7545fc3f86ba8eb70ccce2dff1676),
# file src/klaxon/agent/mcp_client.py: the interface the loop calls its tools through, and the recording and replay of
# their results. Changes for this mini-project:
#   - no MCP transport: Klaxon's client starts its MCP server as a process and speaks JSON-RPC to it; here the three
#     tools run in this package's own process (diagnosis_agent.tools), behind the same ToolClient interface
#   - McpError and McpTimeout are named ToolError and ToolTimeout
#   - the recording's label says where the results come from here (data captured from Klaxon's live stack)
#   - ReplayTools passes on a recorded result's simulated_by
"""How the loop calls its tools: one interface, and the recording and replay of every result.

    ToolClient      what the loop needs: open() lists the tools, call() runs one, close() ends
    RecordedTools   writes every tool result to a file before the loop sees it
    ReplayTools     serves a recorded run's results again, for the same calls in the same order, so that a run can be
                    replayed offline with diagnosis_agent.model.ReplayModel

Which tools may be called is the loop's decision (diagnosis_agent.loop), not this module's.
"""

from __future__ import annotations

import json
import time
from collections import deque
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Protocol

from diagnosis_agent.model import read_records, utc_now

START_TIMEOUT = 30.0  # seconds for the tools to start and list themselves
FORMAT = 1  # of a tool recording's records
LABEL = (
    "Tool results as the loop received them, each written here before the model saw it. They come from the incident"
    " data this repo holds (captured from Klaxon's synthetic payments system, on its live stack), or from fakes in a"
    " test. A result with simulated_by was not read from the data: the harness it names answered in its place."
)


class ToolError(RuntimeError):
    """The tools could not start, or a call broke off without a result."""


class ToolTimeout(ToolError):
    """No answer within the time allowed."""


@dataclass(frozen=True)
class ToolSpec:
    """One tool, as its list describes it."""

    name: str
    title: str
    description: str
    input_schema: dict
    annotations: dict = field(default_factory=dict)

    @property
    def read_only(self) -> bool:
        """Marked read-only and not destructive, both said outright: MCP's defaults are the opposite."""
        return self.annotations.get("readOnlyHint") is True and self.annotations.get("destructiveHint") is False


@dataclass(frozen=True)
class ToolResult:
    text: str
    is_error: bool
    simulated_by: str | None = None  # a harness that answered in place of the data (the missing-evidence case)


class ToolClient(Protocol):
    """What the loop needs from its tools."""

    def open(self, timeout: float = START_TIMEOUT) -> list[ToolSpec]: ...

    def call(self, name: str, arguments: Mapping[str, object], timeout: float) -> ToolResult: ...

    def close(self) -> None: ...


def _append(path: Path, records: list[dict]) -> None:
    """Append records, all or none: a write that fails is cut back off, so no half line is left behind."""
    data = "".join(
        json.dumps(r, ensure_ascii=True, separators=(",", ":"), allow_nan=False) + "\n" for r in records
    ).encode("ascii")
    with open(path, "ab") as out:
        start = out.seek(0, 2)
        try:
            out.write(data)
            out.flush()
        except BaseException:
            out.truncate(start)
            raise


class RecordedTools:
    """A ToolClient that writes every result to a JSON Lines file before passing it on: first the tools as
    listed (a replay rebuilds the same system prompt from them), then each call as it came back."""

    def __init__(self, inner: ToolClient, path: Path | str, header: Mapping | None = None) -> None:
        self.inner = inner
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._header = dict(header or {})
        self._seq = 0

    @property
    def server_info(self) -> dict:
        return getattr(self.inner, "server_info", {})

    def open(self, timeout: float = START_TIMEOUT) -> list[ToolSpec]:
        opened = {"kind": "header", "format": FORMAT, "label": LABEL, "opened_at": utc_now()} | self._header
        try:
            specs = self.inner.open(timeout)
        except ToolError as exc:
            _append(self.path, [opened | {"failed": f"{type(exc).__name__}: {exc}"}])
            raise
        listed = {"tools": [asdict(spec) for spec in specs], "server": self.server_info}
        _append(self.path, [opened | listed])
        return specs

    def call(self, name: str, arguments: Mapping[str, object], timeout: float) -> ToolResult:
        self._seq += 1
        record = {"kind": "tool_call", "seq": self._seq, "tool": name, "arguments": dict(arguments)}
        started = time.monotonic()
        try:
            result = self.inner.call(name, arguments, timeout)
        except ToolError as exc:
            outcome = "timeout" if isinstance(exc, ToolTimeout) else "failed"
            seconds = round(time.monotonic() - started, 3)
            _append(self.path, [record | {"outcome": outcome, "error": str(exc), "seconds": seconds, "at": utc_now()}])
            raise
        seconds = round(time.monotonic() - started, 3)
        record |= {"outcome": "result", "is_error": result.is_error, "text": result.text, "seconds": seconds}
        if result.simulated_by is not None:  # the data was not read: say who answered
            record["simulated_by"] = result.simulated_by
        _append(self.path, [record | {"at": utc_now()}])  # before the loop, and so the model, sees it
        return result

    def close(self) -> None:
        self.inner.close()


def _same_arguments(a: object, b: object) -> bool:
    """Equal as JSON, where true is not 1: compared in a canonical form."""
    return json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


class ReplayTools:
    """A recorded run's tool results, served again: the tools as listed then, and each call's result for the same
    call in the same order. Nothing is called. A different call gets ToolError: the replay has gone another way."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        records = read_records(self.path)
        headers = [r for r in records if r.get("kind") == "header" and "tools" in r]
        if len(headers) != 1:
            raise ValueError(f"{self.path.name} should hold one list of tools, and holds {len(headers)}")
        try:
            self._specs = [ToolSpec(**item) for item in headers[0]["tools"]]
        except TypeError:
            raise ValueError(f"the tools listed in {self.path.name} are damaged or edited") from None
        self.server_info = headers[0].get("server", {})
        self._calls = deque(r for r in records if r.get("kind") == "tool_call")
        for record in self._calls:  # a damaged recording is refused here, before anything is served
            fits = isinstance(record.get("tool"), str) and isinstance(record.get("arguments"), dict)
            if record.get("outcome") == "result":
                fits = fits and isinstance(record.get("text"), str) and isinstance(record.get("is_error"), bool)
            elif record.get("outcome") not in ("timeout", "failed"):
                fits = False
            if not fits:
                raise ValueError(f"a tool call in {self.path.name} is damaged or edited")
        self.served = 0

    @property
    def unused(self) -> int:
        return len(self._calls)

    def open(self, timeout: float = START_TIMEOUT) -> list[ToolSpec]:
        return list(self._specs)

    def call(self, name: str, arguments: Mapping[str, object], timeout: float) -> ToolResult:
        if not self._calls:
            raise ToolError(f"{self.path.name} has no more tool calls: this run went further than the recorded one")
        record = self._calls[0]
        if record["tool"] != name or not _same_arguments(record["arguments"], dict(arguments)):
            raise ToolError(f"{self.path.name} recorded another tool call at this point: the replay went another way")
        self._calls.popleft()
        self.served += 1
        if record["outcome"] == "timeout":
            raise ToolTimeout(f"replayed from {self.path.name}: {record.get('error')}")
        if record["outcome"] == "failed":
            raise ToolError(f"replayed from {self.path.name}: {record.get('error')}")
        return ToolResult(record["text"], record["is_error"], record.get("simulated_by"))

    def close(self) -> None:
        pass
