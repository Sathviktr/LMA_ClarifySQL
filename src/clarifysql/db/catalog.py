"""Schema views of an SQLite database, read directly from the file (works for Spider and our demo database).

* schema_info():       tables, columns (name, type), primary keys, foreign keys and row counts
* create_table_text(): the full schema as CREATE TABLE statements (what "full schema" means in the proposal)
* compact_catalog():   only table and column names, one line per table -- what the policy sees first
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from clarifysql.db.safe_exec import open_readonly


@dataclass
class Column:
    name: str
    type: str
    primary_key: bool = False


@dataclass
class Table:
    name: str
    columns: list[Column] = field(default_factory=list)
    foreign_keys: list[tuple[str, str, str]] = field(default_factory=list)  # (column, ref_table, ref_column)
    rows: int = 0

    def column(self, name: str) -> Column | None:
        low = name.lower()
        return next((c for c in self.columns if c.name.lower() == low), None)


@dataclass
class Schema:
    tables: dict[str, Table]

    def table(self, name: str) -> Table | None:
        low = name.lower()
        return next((t for t in self.tables.values() if t.name.lower() == low), None)


def _quote(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


@lru_cache(maxsize=256)
def schema_info(db: str | Path) -> Schema:
    con = open_readonly(db, guarded=False)  # trusted code: fixed queries, names come from the file itself
    try:
        names = [r[0] for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY rowid")]
        tables = {}
        for name in names:
            cols = [Column(r[1], (r[2] or "").lower(), bool(r[5]))
                    for r in con.execute(f"PRAGMA table_info({_quote(name)})")]
            fks = [(r[3], r[2], r[4] or "") for r in con.execute(f"PRAGMA foreign_key_list({_quote(name)})")]
            rows = con.execute(f"SELECT count(*) FROM {_quote(name)}").fetchone()[0]
            tables[name] = Table(name, cols, fks, rows)
        return Schema(tables)
    finally:
        con.close()


def create_table_text(db: str | Path) -> str:
    con = open_readonly(db, guarded=False)
    try:
        sqls = [r[0] for r in con.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY rowid")]
    finally:
        con.close()
    return "\n".join(s.strip() + ";" for s in sqls if s)


def compact_catalog(db: str | Path) -> str:
    """One line per table: `table(column, column, ...)`. Names only, no types or keys."""
    schema = schema_info(db)
    return "\n".join(f"{t.name}({', '.join(c.name for c in t.columns)})" for t in schema.tables.values())
