"""Paired uncertainty intervals: resample databases, keeping all their episodes together."""

from __future__ import annotations

from collections import defaultdict
import numpy as np
from clarifysql.eval.metrics import macro_f1


def _metric(rows, metric):
    if metric in ("execution_accuracy", "testsuite_accuracy"):
        ans = [r for r in rows if r["kind"] != "unanswerable"]
        key = "correct" if metric == "execution_accuracy" else "testsuite_correct"
        return sum(r[key] for r in ans) / len(ans) if ans else None
    if metric == "decision_macro_f1":
        return macro_f1([r["gold_decision"] for r in rows], [r["decision"] for r in rows])[0]
    raise ValueError(metric)


def paired_bootstrap(baseline: list[dict], seeds: list[list[dict]], metric: str,
                     repeats: int = 5000, seed: int = 2026) -> dict:
    if not baseline or not seeds or repeats < 1:
        raise ValueError("Need nonempty paired runs and a positive repeat count")
    by_id = {r["episode"]: r for r in baseline}
    if len(by_id) != len(baseline):
        raise ValueError("Duplicate baseline episodes")
    ordered = []
    for rows in seeds:
        indexed = {r["episode"]: r for r in rows}
        if len(indexed) != len(rows) or indexed.keys() != by_id.keys():
            raise ValueError("Paired runs must have exactly the same episode IDs")
        for id_, b in by_id.items():
            if any(indexed[id_][k] != b[k] for k in ("db_id", "kind", "gold_decision")):
                raise ValueError("Paired episode metadata differs")
        ordered.append([indexed[id_] for id_ in by_id])
    groups = defaultdict(list)
    for i, row in enumerate(baseline):
        groups[row["db_id"]].append(i)
    if len(groups) < 2:
        raise ValueError("Database bootstrap needs at least two databases")
    blocks = list(groups.values())
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(repeats * 10):
        idx = [i for g in rng.integers(0, len(blocks), size=len(blocks)) for i in blocks[g]]
        b = _metric([baseline[i] for i in idx], metric)
        s = [_metric([rows[i] for i in idx], metric) for rows in ordered]
        if b is None or any(v is None for v in s):
            continue
        diffs.append(float(np.mean(s)) - b)
        if len(diffs) == repeats:
            break
    if len(diffs) != repeats:
        raise ValueError("Too many resamples contain no answerable episodes")
    b = _metric(baseline, metric)
    scores = [_metric(rows, metric) for rows in ordered]
    if b is None or any(v is None for v in scores):
        raise ValueError("No answerable episodes")
    return {"metric": metric, "unit": "database", "databases": len(blocks), "episodes": len(baseline),
            "repeats": repeats, "seed": seed, "baseline": b, "system_seeds": scores,
            "system_mean": float(np.mean(scores)), "difference": float(np.mean(scores)) - b,
            "ci95": np.quantile(diffs, [0.025, 0.975]).tolist()}
