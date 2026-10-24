"""describe_table: columns, types, primary key, foreign keys and row count of one table."""

from __future__ import annotations


def describe_table(ctx, table: str) -> str:
    t = ctx.schema.table(table)
    cols = ", ".join(f"{c.name} {c.type or 'any'}{' PK' if c.primary_key else ''}" for c in t.columns)
    lines = [f"table {t.name} ({t.rows} rows): {cols}"]
    for col, ref_table, ref_col in t.foreign_keys:
        lines.append(f"  {t.name}.{col} -> {ref_table}.{ref_col or '?'}")
    for other in ctx.schema.tables.values():
        for col, ref_table, ref_col in other.foreign_keys:
            if ref_table.lower() == t.name.lower() and other.name != t.name:
                lines.append(f"  referenced by {other.name}.{col}")
    return "\n".join(lines)
