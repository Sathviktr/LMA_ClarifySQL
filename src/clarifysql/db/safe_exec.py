"""Run one SQL query safely (proposal, Section 16).

Five independent layers, so one mistake never becomes a write:
  1. the controller passes the database path; the query text can never choose a file,
  2. the file is opened read-only (SQLite URI `mode=ro`),
  3. a text check allows exactly one statement, and it must start with SELECT or WITH,
  4. an SQLite *authorizer* allows only reading, selecting, functions and recursive CTEs, and denies
     ATTACH, PRAGMA, transactions, schema changes, writes and file/extension functions,
  5. a row cap and a time limit stop runaway queries.
"""

from __future__ import annotations

import re
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path

MAX_ROWS = 1000
TIME_LIMIT_S = 5.0

_ALLOWED_ACTIONS = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION,
                    sqlite3.SQLITE_RECURSIVE}
_DENIED_FUNCTIONS = {"load_extension", "readfile", "writefile", "edit", "fts3_tokenizer"}


@dataclass
class ExecResult:
    status: str  # "ok" | "blocked" | "sql_error" | "timeout"
    columns: list[str] = field(default_factory=list)
    rows: list[tuple] = field(default_factory=list)
    truncated: bool = False
    error: str | None = None
    seconds: float = 0.0

    @property
    def ok(self) -> bool:
        return self.status == "ok"


def strip_comments_and_strings(sql: str) -> str:
    """Replace string literals and quoted names with placeholders and drop comments.

    What remains is the query's "skeleton", where a `;` or a keyword can only be real SQL syntax.
    """
    out = []
    i, n = 0, len(sql)
    while i < n:
        ch = sql[i]
        if sql.startswith("--", i):
            j = sql.find("\n", i)
            i = n if j < 0 else j
            out.append(" ")
        elif sql.startswith("/*", i):
            j = sql.find("*/", i + 2)
            i = n if j < 0 else j + 2
            out.append(" ")
        elif ch in "'\"`[":
            close = "]" if ch == "[" else ch
            j = i + 1
            while j < n:
                if sql[j] == close:
                    if close != "]" and j + 1 < n and sql[j + 1] == close:  # doubled quote = escaped
                        j += 2
                        continue
                    break
                j += 1
            out.append("'s'" if ch == "'" else "_q_")
            i = j + 1
        else:
            out.append(ch)
            i += 1
    return "".join(out)


def check_statement(sql: str) -> str | None:
    """Return None if the text is a single SELECT/WITH statement, else the reason it is refused."""
    skeleton = strip_comments_and_strings(sql).strip()
    if not skeleton:
        return "empty query"
    body = skeleton.rstrip().rstrip(";").rstrip()
    if ";" in body:
        return "more than one statement"
    first = re.match(r"[\s(]*([A-Za-z]+)", body)
    if not first or first.group(1).upper() not in ("SELECT", "WITH"):
        return f"only SELECT or WITH queries are allowed (got {first.group(1).upper() if first else '?'})"
    return None


def _authorizer(action, arg1, arg2, db_name, trigger):
    if action not in _ALLOWED_ACTIONS:
        return sqlite3.SQLITE_DENY
    if action == sqlite3.SQLITE_FUNCTION and (arg2 or "").lower() in _DENIED_FUNCTIONS:
        return sqlite3.SQLITE_DENY
    return sqlite3.SQLITE_OK


def open_readonly(path: str | Path, guarded: bool = True) -> sqlite3.Connection:
    """Open an SQLite file read-only. `guarded=False` skips the authorizer (trusted tool code only)."""
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    con = sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True, check_same_thread=False)
    con.text_factory = lambda b: b.decode("utf-8", errors="replace")
    if guarded:
        con.set_authorizer(_authorizer)
    return con


def execute(db: str | Path, sql: str, max_rows: int = MAX_ROWS, time_limit_s: float = TIME_LIMIT_S) -> ExecResult:
    """Run one query against the database file `db` under all safety layers."""
    t0 = time.perf_counter()
    reason = check_statement(sql)
    if reason:
        return ExecResult("blocked", error=reason, seconds=time.perf_counter() - t0)
    con = open_readonly(db)
    deadline = t0 + time_limit_s
    con.set_progress_handler(lambda: 1 if time.perf_counter() > deadline else 0, 10_000)
    try:
        cur = con.execute(sql)
        rows = cur.fetchmany(max_rows + 1)
        columns = [d[0] for d in (cur.description or [])]
        truncated = len(rows) > max_rows
        return ExecResult("ok", columns, [tuple(r) for r in rows[:max_rows]], truncated,
                          seconds=time.perf_counter() - t0)
    except sqlite3.DatabaseError as exc:
        msg = str(exc)
        if "interrupted" in msg:
            status = "timeout"
        elif "not authorized" in msg or "prohibited" in msg or "readonly" in msg.replace("-", ""):
            status = "blocked"
        else:
            status = "sql_error"
        return ExecResult(status, error=msg, seconds=time.perf_counter() - t0)
    except (sqlite3.Warning, sqlite3.ProgrammingError) as exc:  # e.g. "one statement at a time"
        return ExecResult("blocked", error=str(exc), seconds=time.perf_counter() - t0)
    finally:
        con.close()
