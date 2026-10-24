"""Tool specifications, argument checking and dispatch.

Every tool call proposed by the policy is checked *before* anything runs (proposal, Section 16):
known tool, no unknown or missing arguments, correct types, numbers inside their range, and table and
column names that really exist in this database (matched case-insensitively). A rejected call returns a
helpful error (with close matches), so the policy can recover on its next step.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from clarifysql.db.catalog import Schema, schema_info
from clarifysql.tools.describe_table import describe_table
from clarifysql.tools.sample_values import sample_values
from clarifysql.tools.search_definitions import Definition, search_definitions

TOOL_SPECS: dict[str, dict] = {
    "describe_table": {
        "description": "Columns, types, keys and row count of one table.",
        "parameters": {"table": {"type": "string"}},
        "required": ["table"],
    },
    "sample_values": {
        "description": "The most frequent values of one column, with counts.",
        "parameters": {"table": {"type": "string"}, "column": {"type": "string"},
                       "limit": {"type": "integer", "minimum": 1, "maximum": 10, "default": 5}},
        "required": ["table", "column"],
    },
    "search_definitions": {
        "description": "Search the glossary for the meaning of a vague phrase (e.g. 'high mark').",
        "parameters": {"query": {"type": "string", "maxLength": 200},
                       "k": {"type": "integer", "minimum": 1, "maximum": 5, "default": 3}},
        "required": ["query"],
    },
}
_FUNCS = {"describe_table": describe_table, "sample_values": sample_values,
          "search_definitions": search_definitions}
_TYPES = {"string": str, "integer": int}


@dataclass
class ToolContext:
    """What the controller binds for one episode: the database file and its glossary."""

    db_path: Path
    glossary: list[Definition] = field(default_factory=list)

    @property
    def schema(self) -> Schema:
        return schema_info(str(self.db_path))


@dataclass
class ToolResult:
    ok: bool
    text: str
    tool: str
    args: dict[str, Any]


def _suggest(name: str, options: list[str]) -> str:
    close = difflib.get_close_matches(name, options, n=3, cutoff=0.5)
    return f" Did you mean: {', '.join(close)}?" if close else ""


def validate_call(tool: Any, args: Any, schema: Schema) -> tuple[dict | None, str | None]:
    """Return (cleaned arguments, None) for a valid call, or (None, error message)."""
    if not isinstance(tool, str) or tool not in TOOL_SPECS:
        return None, f"unknown tool {tool!r}; available: {', '.join(TOOL_SPECS)}"
    if not isinstance(args, dict):
        return None, "arguments must be a JSON object"
    spec = TOOL_SPECS[tool]
    params = spec["parameters"]
    unknown = sorted(set(args) - set(params))
    if unknown:
        return None, f"unknown argument(s) for {tool}: {', '.join(unknown)}"
    missing = [p for p in spec["required"] if p not in args]
    if missing:
        return None, f"missing argument(s) for {tool}: {', '.join(missing)}"
    clean: dict[str, Any] = {}
    for name, p in params.items():
        if name not in args:
            if "default" in p:
                clean[name] = p["default"]
            continue
        value = args[name]
        expected = _TYPES[p["type"]]
        if not isinstance(value, expected) or isinstance(value, bool):
            article = "an" if p["type"][0] in "aeiou" else "a"
            return None, f"{name} must be {article} {p['type']}"
        if expected is int and not (p["minimum"] <= value <= p["maximum"]):
            return None, f"{name} must be between {p['minimum']} and {p['maximum']}"
        if expected is str:
            value = value.strip()
            if not value or len(value) > p.get("maxLength", 128):
                return None, f"{name} must be a non-empty string of at most {p.get('maxLength', 128)} characters"
        clean[name] = value
    if "table" in clean:
        table = schema.table(clean["table"])
        if table is None:
            return None, f"no table named {clean['table']!r}.{_suggest(clean['table'], list(schema.tables))}"
        clean["table"] = table.name
        if "column" in clean:
            col = table.column(clean["column"])
            if col is None:
                names = [c.name for c in table.columns]
                return None, (f"table {table.name} has no column {clean['column']!r}."
                              f"{_suggest(clean['column'], names)}")
            clean["column"] = col.name
    return clean, None


def run_tool(ctx: ToolContext, tool: Any, args: Any) -> ToolResult:
    clean, error = validate_call(tool, args, ctx.schema)
    if error:
        return ToolResult(False, f"rejected: {error}", str(tool), args if isinstance(args, dict) else {})
    return ToolResult(True, _FUNCS[tool](ctx, **clean), tool, clean)
