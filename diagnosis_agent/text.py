# Copied from Klaxon (github.com/AravindKumar2504/klaxon) at commit fef450b (fef450b0aae7545fc3f86ba8eb70ccce2dff1676),
# file src/klaxon/agent/tools.py: only its cleaning and its argument checks, the parts that need no live stack.
# Changes for this mini-project:
#   - only these parts are copied; Klaxon's tools module also reads Prometheus, Loki, Tempo, docker and PostgreSQL,
#     and imports httpx, psycopg and confluent_kafka, none of which this package needs
#   - _check_value also takes "enum" (search_logs here names its services and levels; Klaxon's took LogQL text)
"""Cleaning text that leaves a tool or a model, and checking a tool's arguments against its input schema.

Everything a tool returns comes from the system under investigation, so it is untrusted: a log line can say anything,
"ignore your instructions" included. Every result and every error message is cleaned before it leaves: control and
invisible characters are dropped (variation selectors and private-use characters too, which can hide bytes), this
Mac's home folder becomes ~, and every string and list is capped.
"""

from __future__ import annotations

import json
import math
import re
import unicodedata
from pathlib import Path

MAX_STRING = 1000  # characters in any one string that leaves a tool; most fields are cut far shorter
MAX_FIELD = 300  # a label, an attribute, a status
MAX_LIST = 100  # every list cap is at most this, so clean() never cuts a list a tool already counted
MAX_KEYS = 40
MAX_DEPTH = 8
BUDGET = 30_000  # characters of JSON for a tool's longest list
HOME = str(Path.home())


class ToolInputError(ValueError):
    """The arguments break the tool's schema. The model can correct it."""


def _hidden(ch: str) -> bool:
    """Characters that do not show, or show as something else: control, format, private-use, unassigned."""
    if ch in "\n\t":
        return False
    if 0xFE00 <= ord(ch) <= 0xFE0F or 0xE0100 <= ord(ch) <= 0xE01EF:  # variation selectors: can carry hidden bytes
        return True
    return unicodedata.category(ch) in ("Cc", "Cf", "Cs", "Co", "Cn")


def clean_text(value: str, limit: int = MAX_STRING) -> str:
    """Drop hidden characters (newline and tab stay), write this Mac's home folder as ~, then cap the length."""
    kept = "".join(ch for ch in value if not _hidden(ch))
    if len(HOME) > 1:
        kept = kept.replace(HOME, "~")
    if len(kept) > limit:
        return f"{kept[:limit]}...[{len(kept) - limit} more characters]"
    return kept


def clean(value: object, depth: int = 0) -> object:
    """A JSON-able copy with every string, list and object capped."""
    if depth > MAX_DEPTH:
        return "...[nested too deeply]"
    if value is None or isinstance(value, bool | int):
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        if value.is_integer() or abs(value) >= 1e6:  # a count stays exact
            return round(value, 3)
        return float(f"{value:.6g}")
    if isinstance(value, str):
        return clean_text(value)
    if isinstance(value, dict):
        items = list(value.items())
        out = {clean_text(str(key), 100): clean(item, depth + 1) for key, item in items[:MAX_KEYS]}
        if len(items) > MAX_KEYS:
            out["..."] = f"{len(items) - MAX_KEYS} more keys"
        return out
    if isinstance(value, list | tuple):
        out = [clean(item, depth + 1) for item in value[:MAX_LIST]]
        if len(value) > MAX_LIST:
            out.append(f"...[{len(value) - MAX_LIST} more items]")
        return out
    return clean_text(repr(value), MAX_FIELD)


def _size(item: object) -> int:
    return len(json.dumps(clean(item), ensure_ascii=False)) + 2


def fit(items: list, budget: int = BUDGET) -> list:
    """The first items that fit in the budget (at least one, so an answer is never empty for size alone)."""
    out, used = [], 0
    for item in items:
        used += _size(item)
        if out and used > budget:
            break
        out.append(item)
    return out


# Arguments: the subset of JSON Schema the tools' input schemas use.


def check_arguments(schema: dict, arguments: object) -> dict:
    """The arguments, checked against the tool's input schema, with defaults filled in."""
    if arguments is None:
        arguments = {}
    if not isinstance(arguments, dict):
        raise ToolInputError("arguments must be a JSON object")
    properties = schema.get("properties", {})
    unknown = sorted(str(name) for name in set(arguments) - set(properties))
    if unknown:
        takes = ", ".join(properties) or "no arguments"
        raise ToolInputError(f"unknown argument(s): {clean_text(', '.join(unknown), 200)}. This tool takes: {takes}")
    out = {}
    for name, rule in properties.items():
        if name in arguments:
            out[name] = _check_value(name, rule, arguments[name])
        elif name in schema.get("required", ()):
            raise ToolInputError(f"{name} is required")
        elif "default" in rule:
            out[name] = rule["default"]
    return out


def _check_value(name: str, rule: dict, value: object) -> object:
    kind = rule["type"]
    if kind == "string":
        if not isinstance(value, str):
            raise ToolInputError(f"{name} must be a string")
        if "enum" in rule and value not in rule["enum"]:
            raise ToolInputError(f"{name} must be one of: {', '.join(rule['enum'])}")
        if len(value) < rule.get("minLength", 0):
            raise ToolInputError(f"{name} must not be empty")
        if len(value) > rule.get("maxLength", MAX_STRING):
            raise ToolInputError(f"{name} is longer than {rule.get('maxLength', MAX_STRING)} characters")
        # Every pattern here is written to match the whole value (^...$). fullmatch, not search: in Python, $ also
        # matches just before a trailing newline, so search would let "ab\n" through.
        if "pattern" in rule and re.fullmatch(rule["pattern"], value) is None:
            raise ToolInputError(f"{name} does not match {rule['pattern']}")
        return value
    if kind == "integer":
        if isinstance(value, float) and value.is_integer():  # JSON Schema counts 15.0 as an integer
            value = int(value)
        if not isinstance(value, int) or isinstance(value, bool):
            raise ToolInputError(f"{name} must be a whole number")
        if not rule.get("minimum", value) <= value <= rule.get("maximum", value):
            raise ToolInputError(f"{name} must be from {rule.get('minimum')} to {rule.get('maximum')}")
        return value
    if kind == "boolean":
        if not isinstance(value, bool):
            raise ToolInputError(f"{name} must be true or false")
        return value
    raise AssertionError(f"schema type {kind} is not supported")  # a bug in a schema, not bad input
