"""Check the data facts that the proposal states (Section 8.1) against the downloaded files.

Usage:  python -I scripts/data_stats.py
Writes results/data/data_stats.json and prints a claim-by-claim comparison.
"""

from __future__ import annotations

import json
from collections import Counter

from clarifysql.config import results_dir
from clarifysql.data import cosql, spider, trustsql
from clarifysql.data.normalize import normalize_sql
from clarifysql.db.safe_exec import execute


def main() -> None:
    stats: dict = {"spider": {}, "cosql": {}, "trustsql": {}}
    sp = stats["spider"]
    dbs_by_split = {}
    for split in ("train", "train_others", "dev", "test"):
        ex = spider.load_split(split)
        dbs_by_split[split] = set(spider.databases(split))
        sp[split] = {"questions": len(ex), "databases": len(dbs_by_split[split]),
                     "distinct_queries": len({(e.db_id, normalize_sql(e.query)) for e in ex}),
                     "distinct_queries_ignoring_literal_case":
                         len({(e.db_id, normalize_sql(e.query).lower()) for e in ex})}
    overlaps = {f"{a}&{b}": len(dbs_by_split[a] & dbs_by_split[b])
                for a in dbs_by_split for b in dbs_by_split if a < b}
    sp["database_overlaps"] = overlaps

    status, empty, slow = Counter(), 0, []
    for e in spider.load_split("dev"):
        r = execute(spider.db_path(e.db_id), e.query)
        status[r.status] += 1
        if r.ok and not r.rows:
            empty += 1
        if r.seconds > 1.0:
            slow.append((e.idx, round(r.seconds, 2)))
    n_dev = sp["dev"]["questions"]
    sp["dev_gold_execution"] = {"status_counts": dict(status), "empty_results": empty,
                                "empty_percent": round(100 * empty / n_dev, 1), "slow_over_1s": slow}

    co = stats["cosql"]
    for split in ("train", "dev"):
        c = cosql.intent_counts(split)
        co[split] = {"user_turns": len(cosql.load_intents(split)), "AMBIGUOUS": c["AMBIGUOUS"],
                     "CANNOT_ANSWER": c["CANNOT_ANSWER"]}

    inf = trustsql.load_infeasible()
    stats["trustsql"] = {"infeasible": len(inf), "by_note": dict(Counter(r["note"] for r in inf)),
                         "missing_column": len(trustsql.missing_column_questions()),
                         "databases": sorted({r["db_id"] for r in inf}),
                         "all_on_spider_dev": {r["db_id"] for r in inf} <= dbs_by_split["dev"]}

    claims = [
        ("Spider core train: 7,000 questions", sp["train"]["questions"], 7000),
        ("... on 140 databases", sp["train"]["databases"], 140),
        ("legacy train_others: 1,659 questions", sp["train_others"]["questions"], 1659),
        ("... on 6 databases", sp["train_others"]["databases"], 6),
        ("dev: 1,034 questions", n_dev, 1034),
        ("... on 20 databases", sp["dev"]["databases"], 20),
        ("test: 2,147 questions", sp["test"]["questions"], 2147),
        ("... on 40 databases", sp["test"]["databases"], 40),
        ("no database in two splits", sum(overlaps.values()), 0),
        ("dev: 551 distinct queries (ignoring literal case)", sp["dev"]["distinct_queries_ignoring_literal_case"], 551),
        ("all dev gold queries run read-only", status["ok"], n_dev),
        ("4.7% of dev gold queries return no rows", sp["dev_gold_execution"]["empty_percent"], 4.7),
        ("CoSQL train AMBIGUOUS 875", co["train"]["AMBIGUOUS"], 875),
        ("CoSQL train CANNOT_ANSWER 184", co["train"]["CANNOT_ANSWER"], 184),
        ("CoSQL dev AMBIGUOUS 107", co["dev"]["AMBIGUOUS"], 107),
        ("CoSQL dev CANNOT_ANSWER 20", co["dev"]["CANNOT_ANSWER"], 20),
        ("TrustSQL infeasible (Spider test): 527", stats["trustsql"]["infeasible"], 527),
        ("... of which missing-column: 300", stats["trustsql"]["missing_column"], 300),
    ]
    stats["claims"] = [{"claim": c, "measured": m, "proposal": p, "match": m == p} for c, m, p in claims]
    out = results_dir() / "data" / "data_stats.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    width = max(len(c) for c, _, _ in claims)
    for c, m, p in claims:
        print(f"{'OK ' if m == p else 'DIFF'} {c:<{width}}  measured={m}  proposal={p}")
    print(f"\ndev distinct queries keeping literal case (used for de-duplication): {sp['dev']['distinct_queries']}")
    print(f"written to {out}")


if __name__ == "__main__":
    main()
