"""Automatic checks for episodes (proposal, Section 8.3: "the SQL runs, the vague wording really hides the
detail, the reply resolves it"), plus retrieval and leakage checks.

An episode passes when check_episode() returns no problems. Weak items are dropped rather than kept to
reach a target size.
"""

from __future__ import annotations

import re
from collections import Counter

from clarifysql.data import spider
from clarifysql.db.catalog import schema_info
from clarifysql.db.safe_exec import execute
from clarifysql.episodes.schema import Episode
from clarifysql.tools.search_definitions import Definition, search, tokenize

_MEASURE_STOP = {"number", "amount", "count", "total", "average", "of", "the", "most", "times"}


def _numbers(text: str) -> set[str]:
    return {n.rstrip("0").rstrip(".") if "." in n else n for n in re.findall(r"-?\d+(?:\.\d+)?", text)}


def check_episode(ep: Episode) -> list[str]:
    problems = []
    db = spider.db_path(ep.db_id)
    if ep.kind == "unanswerable":
        if ep.gold_sql is not None:
            problems.append("unanswerable episode has gold SQL")
        if ep.draft.get("method") == "template":
            words = {w.lower() for t in schema_info(str(db)).tables.values() for c in t.columns
                     for w in re.split(r"[_\s]+", c.name)}
            if set(tokenize(ep.missing["attribute"])) & {tokenize(w)[0] for w in words if tokenize(w)}:
                problems.append("the 'missing' attribute looks like an existing column")
    elif not execute(db, ep.gold_sql).ok:
        problems.append("gold SQL does not run")

    if ep.kind in ("ambiguous", "two_slot", "definition"):
        if ep.draft.get("needs_writing"):
            problems.append("request still has to be written by the team")
        elif ep.request.strip().lower() == (ep.source_question or "").strip().lower():
            problems.append("request equals the original question")
        for s in ep.slots:
            if s["kind"] == "threshold":
                if _numbers(s["value"]) & _numbers(ep.request):
                    problems.append("the request still states the hidden value")
                if not _numbers(s["value"]) <= _numbers(s["answer"]):
                    problems.append("the user's answer does not give the hidden value")
            else:
                measure = [w for w in tokenize(s["value"]) if w not in _MEASURE_STOP]
                if measure and set(measure) <= set(tokenize(ep.request)):
                    problems.append("the request still names the hidden measure")
                if len(s["meanings"]) < 2:
                    problems.append("a ranking slot needs at least two possible measures")
        glossary = [Definition(g["term"], g["definition"], tuple(g.get("aliases", []))) for g in ep.glossary]
        terms = {s["term"] for s in ep.slots}
        if ep.kind == "definition":
            hits = search(glossary, ep.reference[0]["args"]["query"], k=1)
            if not hits or hits[0][0].term not in terms:
                problems.append("the defining glossary entry is not retrieved first")
        elif any(g.term in terms for g in glossary):
            problems.append("an ambiguous slot is defined in the glossary")
        visible = (ep.request + " " + " ".join(g.text for g in glossary)).lower()
        if ep.kind != "definition" and any(s["hidden"] and s["hidden"].lower() in visible for s in ep.slots):
            problems.append("the hidden detail is visible to the policy")
    return problems


def check_episodes(episodes: list[Episode]) -> dict:
    per = {ep.id: check_episode(ep) for ep in episodes}
    counts = Counter(p for probs in per.values() for p in probs)
    ok = sum(not probs for probs in per.values())
    return {"summary": {"episodes": len(episodes), "passed": ok, "problems": dict(counts)}, "per_episode": per}
