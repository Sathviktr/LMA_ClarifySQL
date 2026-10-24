"""sample_values: the most frequent non-empty values of one column, with how often each occurs.

Useful when a word in the request might be a stored value ("France", "Spring Gala"), or when the agent
needs to see what a column's values look like before asking or executing.
"""

from __future__ import annotations

from clarifysql.db.safe_exec import execute


def _q(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def sample_values(ctx, table: str, column: str, limit: int = 5) -> str:
    t = ctx.schema.table(table)
    c = t.column(column)
    sql = (f"SELECT {_q(c.name)}, count(*) AS n FROM {_q(t.name)} WHERE {_q(c.name)} IS NOT NULL "
           f"GROUP BY {_q(c.name)} ORDER BY n DESC, {_q(c.name)} LIMIT {int(limit)}")
    res = execute(ctx.db_path, sql)  # built from validated names, and still run through every safety layer
    if not res.ok:
        return f"could not read {t.name}.{c.name}: {res.error}"
    if not res.rows:
        return f"{t.name}.{c.name} has no values"
    values = ", ".join(f"{v!r} ({n})" for v, n in res.rows)
    return f"{t.name}.{c.name} most frequent values: {values}"
