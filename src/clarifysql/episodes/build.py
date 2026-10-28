"""Build the episodes of one split (proposal, Sections 8.2 and 8.3).

Sources: Spider questions of the split's databases (clear, ambiguous, two-slot and definition episodes),
templates on the split's own databases (unanswerable, train and validation) and TrustSQL's hand-written
missing-column questions (unanswerable, test). Every random choice uses a fixed seed.
"""

from __future__ import annotations

import math
import random
import re
from collections import defaultdict

from clarifysql.data import spider, trustsql
from clarifysql.data.normalize import normalize_sql
from clarifysql.data.splits import load_splits
from clarifysql.db.catalog import schema_info
from clarifysql.episodes.ambiguate import Slot, candidate_measures, ranking_draft, threshold_draft
from clarifysql.episodes.checks import clashes
from clarifysql.episodes.mining import dedupe, is_ranking, is_threshold
from clarifysql.episodes.schema import Episode

SEED = 2026
TARGETS = {
    "train": {"clear": 130, "ambiguous": 120, "two_slot": 15, "definition": 30, "unanswerable": 45},
    "validation": {"clear": 24, "ambiguous": 16, "two_slot": 2, "definition": 6, "unanswerable": 12},
    "test": {"clear": 80, "ambiguous": 56, "two_slot": 8, "definition": 16, "unanswerable": 40},
}
MAX_DISTRACTORS = 4

PERSON_ATTRIBUTES = [
    "email address", "phone number", "salary", "nickname", "birthday", "favorite color", "annual income",
    "marital status", "blood type", "religion", "credit score", "hobby", "number of children", "eye color",
    "social media account", "home address", "spoken languages",
]
THING_ATTRIBUTES = [
    "website", "carbon footprint", "customer satisfaction score", "logo color", "opening hours", "official motto",
    "number of social media followers", "environmental rating", "insurance provider", "energy consumption",
]
PERSON_WORDS = {"people", "person", "student", "employee", "customer", "user", "member", "staff", "player", "singer",
                "actor", "author", "driver", "teacher", "manager", "head", "pilot", "artist", "musician", "climber",
                "professor", "instructor", "physician", "nurse", "patient", "candidate", "voter", "host", "owner",
                "participant", "representative", "editor", "journalist", "perpetrator", "gymnast", "wrestler",
                "conductor", "director", "faculty", "captain", "visitor", "guest", "investor", "entrepreneur"}
PERSON_COLUMNS = {"first_name", "last_name", "fname", "lname", "age", "sex", "gender", "date_of_birth", "birth_date"}


def is_person_table(table) -> bool:
    words = {singular(w) for w in humanize(table.name).split()}
    cols = {c.name.lower() for c in table.columns}
    return bool(words & PERSON_WORDS) or len(cols & PERSON_COLUMNS) >= 2
UNANSWERABLE_TEMPLATES = [
    "What is the {attr} of each {entity}?",
    "List all {entities} together with their {attr}.",
    "Which {entity} has the highest {attr}?",
    "Show the average {attr} of the {entities}.",
]


# ----------------------------------------------------------------------------------------- naming helpers
def humanize(name: str) -> str:
    name = re.sub(r"([a-z])([A-Z])", r"\1 \2", name.replace("_", " "))
    return re.sub(r"\s+", " ", name).strip().lower()


def singular(word: str) -> str:
    w = humanize(word)
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if w.endswith(("oes", "ses", "xes", "ches", "shes")) and len(w) > 4:
        return w[:-2]
    if w.endswith("s") and not w.endswith("ss") and len(w) > 3:
        return w[:-1]
    return w


def plural(word: str) -> str:
    w = singular(word)
    if w.endswith("y") and not w.endswith(("ay", "ey", "oy", "uy")):
        return w[:-1] + "ies"
    return w + ("es" if w.endswith(("s", "x", "ch", "sh")) else "s")


def alias_map(sql: str) -> dict[str, str]:
    return {a.lower(): t for t, a in re.findall(r"(?:from|join)\s+(\w+)\s+(?:as\s+)?(t\d+)\b", sql, re.I)}


def slot_table(target: str, sql: str, schema) -> str | None:
    """The table a slot's SQL expression belongs to (resolving aliases like T2.age)."""
    m = re.match(r"(?:(\w+)\.)?(\w+)$", target.strip())
    tables = [t for t in schema.tables if re.search(rf"\b{re.escape(t)}\b", sql, re.I)]
    if m:
        prefix, col = m.groups()
        if prefix:
            real = alias_map(sql).get(prefix.lower(), prefix)
            if schema.table(real):
                return schema.table(real).name
        for t in tables:
            if schema.tables[t].column(col):
                return t
    return tables[0] if tables else None


def column_phrase(target: str) -> str:
    if re.search(r"count\s*\(", target, re.I):
        return "number"
    return humanize(target.split(".")[-1].strip("() "))


def following_noun(request: str, word: str) -> str | None:
    m = re.search(rf"\b{re.escape(word)}\s+([A-Za-z]+)", request)
    stop = {"and", "or", "but", "in", "of", "on", "at", "by", "for", "with", "from", "to", "the", "a", "an", "that",
            "who", "which", "whose", "is", "are", "was", "were", "than", "if"}
    return m.group(1) if m and m.group(1).lower() not in stop else None


# --------------------------------------------------------------------------------- slots -> text pieces
_ADJ = {"high", "low", "large", "big", "small", "long", "short", "tall", "heavy"}


def describe_slot(slot: Slot | dict, request: str, entity: str, source_question: str = "") -> dict:
    """Readable pieces of a slot: the glossary term, its definition, aliases and the clarifying question.

    term        the phrase a policy would search for ("older head", "many airports", "city ranks highest")
    definition  what the term means in the hidden intent ("age older than 56", "ranked by population")
    question    a short, targeted clarifying question (the reference target for the policy)
    """
    s = slot if isinstance(slot, dict) else slot.to_dict()
    vague, col = s["vague"], column_phrase(s["target"])
    if s["kind"] == "ranking":
        m = re.search(r"\b([A-Za-z]+)\s+(?:that\s+|who\s+|which\s+)?ranks\s+(?:highest|lowest)", request)
        if m and m.group(1).lower() not in {"that", "who", "which", "the", "it", "one"}:
            entity = singular(m.group(1))
        options = [humanize(m.split(".")[-1]) if "." in m else m for m in s["meanings"]][:3]
        listed = ", ".join(options[:-1]) + f" or {options[-1]}" if len(options) >= 2 else "which measure"
        return {"term": f"{entity} {vague}", "definition": f"ranked by {s['value']}", "aliases": [],
                "question": f"Ranked by what: {listed}?"}
    noun = None
    if source_question and s["hidden"] and s["hidden"] in source_question:
        noun = following_noun(source_question[source_question.index(s["hidden"]):], s["hidden"].split()[-1])
    noun = noun or following_noun(request, vague)
    if vague in ("many", "few"):
        thing = noun or plural(entity)
        return {"term": f"{vague} {thing}", "definition": f"{s['hidden']} {thing}", "aliases": [],
                "question": f'How many {thing} count as "{vague}"?',
                "meanings": _threshold_meanings(s["value"], f"number of {thing}")}
    if vague in ("older", "younger"):
        return {"term": f"{vague} {entity}", "definition": f"{col} {s['hidden']}", "aliases": [],
                "question": f'What {col} counts as "{vague}"?', "meanings": _threshold_meanings(s["value"], col)}
    if vague in ("recently", "long ago"):
        adj = "recent" if vague == "recently" else "early"
        return {"term": f"{adj} {entity}", "definition": f"{col} {s['hidden']}", "aliases": [f"{entity} {vague}"],
                "question": f'Which {col} do you mean by "{vague}"?', "meanings": _threshold_meanings(s["value"], col)}
    if vague == "within a certain range":
        return {"term": f"{col} range", "definition": f"{col} {s['hidden']}", "aliases": [],
                "question": f"Which range of {col} do you mean?", "meanings": _threshold_meanings(s["value"], col)}
    if vague in _ADJ:
        nxt = following_noun(request, vague)
        word = humanize(nxt) if (nxt and col == "number") else col  # a count: use the noun the request says
        return {"term": f"{vague} {word}", "definition": f"{word} {s['hidden']}", "aliases": [],
                "question": f'What {word} counts as "{vague}"?', "meanings": _threshold_meanings(s["value"], word)}
    return {"term": f"{vague} {entity}", "definition": f"{col} {s['hidden']}", "aliases": [],
            "question": f'What do you mean by "{vague}"?', "meanings": _threshold_meanings(s["value"], col)}


def _threshold_meanings(value: str, subject: str) -> list[str]:
    """Readable form of a threshold slot's open question, e.g. 'number of customers above some value (which?)'."""
    op = value.split()[0] if value else ""
    if op == "between":
        return [f"{subject} between two values (which?)"]
    word = {">": "above", ">=": "at least", "<": "below", "<=": "at most"}.get(op, "compared with")
    return [f"{subject} {word} some value (which?)"]


# ----------------------------------------------------------------------------------- candidate records
def _ambiguity_candidates(examples, schemas) -> list[dict]:
    out = []
    for e in examples:
        if is_threshold(e):
            d = threshold_draft(e.question, e.query)
            if d:
                out.append({"ex": e, "request": d[0], "slots": [d[1]], "kind": "threshold"})
                continue
        if is_ranking(e):
            r = ranking_draft(e.question, e.query, candidate_measures(e.query, schemas[e.db_id]))
            if r and len(r[1].meanings) >= 2:
                out.append({"ex": e, "request": r[0], "slots": [r[1]], "kind": "ranking"})
    return out


def _two_slot(c: dict, schemas) -> dict | None:
    """Hide a second detail in an ambiguity candidate: a second cut-off, or the ranking measure."""
    e, first = c["ex"], c["slots"][0]
    if c["request"] is None:
        return None
    if first.kind == "threshold":
        d2 = threshold_draft(c["request"], e.query)
        if d2 and d2[1].value != first.value:
            return {"ex": e, "request": d2[0], "slots": [first, d2[1]], "kind": "two_slot"}
        if is_ranking(e):
            r = ranking_draft(c["request"], e.query, candidate_measures(e.query, schemas[e.db_id]))
            if r and r[0] and len(r[1].meanings) >= 2:
                return {"ex": e, "request": r[0], "slots": [first, r[1]], "kind": "two_slot"}
    return None


def _round_robin(items: list, key, rng: random.Random, n: int) -> list:
    """Pick n items spread over databases: shuffle within each database, then take one per database in turn."""
    groups = defaultdict(list)
    for it in items:
        groups[key(it)].append(it)
    for g in groups.values():
        rng.shuffle(g)
    order = sorted(groups)
    rng.shuffle(order)
    picked = []
    while len(picked) < n and any(groups.values()):
        for k in order:
            if groups[k] and len(picked) < n:
                picked.append(groups[k].pop())
    return picked


# ------------------------------------------------------------------------------------------- the builder
def build_split(split: str, seed: int = SEED) -> tuple[list[Episode], dict]:
    from clarifysql.episodes.checks import check_episode

    rng = random.Random(f"{split}-{seed}")
    target = TARGETS[split]
    source = "dev" if split == "test" else "train"
    dbs = set(load_splits()[split])
    unique = dedupe(e for e in spider.load_split(source) if e.db_id in dbs)
    schemas = {db: schema_info(str(spider.db_path(db))) for db in dbs}
    review = split in ("validation", "test")
    spare = 1.0 if review else 1.4  # training: build spares, then keep only episodes that pass every check

    def key(c):
        return c["ex"].db_id

    cands = _ambiguity_candidates(unique, schemas)
    usable = [c for c in cands if c["request"] is not None or review]
    two = [t for c in usable if (t := _two_slot(c, schemas))]
    picked_two = _round_robin(two, key, rng, math.ceil(target["two_slot"] * spare))
    used = {id(c["ex"]) for c in picked_two}
    rest = [c for c in usable if id(c["ex"]) not in used]
    by_kind = {k: [c for c in rest if c["kind"] == k] for k in ("threshold", "ranking")}
    n_def = math.ceil(target["definition"] * spare)
    n_amb = n_def + math.ceil(target["ambiguous"] * spare)
    half = n_amb // 2
    pool = _round_robin(by_kind["threshold"], key, rng, half) + _round_robin(by_kind["ranking"], key, rng, n_amb - half)
    if len(pool) < n_amb:  # top up from whichever kind has more left
        left = [c for c in rest if c not in pool]
        pool += _round_robin(left, key, rng, n_amb - len(pool))
    rng.shuffle(pool)
    definition_src, ambiguous_src = pool[:n_def], pool[n_def:]

    taken = {(c["ex"].db_id, normalize_sql(c["ex"].query)) for c in pool + picked_two}
    clear_pool = [e for e in unique if (e.db_id, normalize_sql(e.query)) not in taken]
    clear_src = _round_robin(clear_pool, lambda e: e.db_id, rng, target["clear"])

    # glossary entries per database, from every ambiguity candidate of the split (used as distractors)
    entries_by_db = defaultdict(list)
    for c in cands:
        e = c["ex"]
        for sl in c["slots"]:
            entity = singular(slot_table(sl.target, e.query, schemas[e.db_id]) or "record")
            desc = describe_slot(sl, c["request"] or e.question, entity, e.question)
            entries_by_db[e.db_id].append({"term": desc["term"], "definition": desc["definition"],
                                           "aliases": desc["aliases"], "source_id": f"{source}-{e.idx}",
                                           "hidden": sl.hidden})

    def glossary_for(db: str, own_terms: set[str], own_hidden: list[str], own_source: str,
                     include: list[dict]) -> list[dict]:
        own_numbers = {n for h in own_hidden for n in re.findall(r"-?\d+(?:\.\d+)?", h or "")}
        seen, others = set(own_terms), []
        for g in entries_by_db[db]:
            if g["source_id"] == own_source or g["term"] in seen or clashes(g["term"], own_terms):
                continue
            if any(h and h.lower() in g["definition"].lower() for h in own_hidden):
                continue
            if own_numbers & set(re.findall(r"-?\d+(?:\.\d+)?", g["definition"])):
                continue  # no distractor may state the hidden number, even for another term
            seen.add(g["term"])
            others.append(g)
        pick = random.Random(own_source).sample(others, min(MAX_DISTRACTORS, len(others)))
        out = include + pick
        random.Random(own_source + "x").shuffle(out)
        return [{k: g[k] for k in ("term", "definition", "aliases")} for g in out]

    def make_vague(c: dict, kind: str) -> Episode:
        e = c["ex"]
        sid = f"{source}-{e.idx}"
        request = c["request"] or e.question
        slots = []
        for sl in c["slots"]:
            d = sl.to_dict()
            entity = singular(slot_table(sl.target, e.query, schemas[e.db_id]) or "record")
            d.update(describe_slot(d, request, entity, e.question))
            slots.append(d)
        include = ([{k: slots[0][k] for k in ("term", "definition", "aliases")}] if kind == "definition" else [])
        glossary = glossary_for(e.db_id, {d["term"] for d in slots}, [d["hidden"] for d in slots], sid, include)
        ref = [{"action": "inspect", "tool": "search_definitions", "args": {"query": slots[0]["term"]}}]
        if kind != "definition":
            ref += [{"action": "clarify", "question": d["question"],
                     "slots": [{"name": d["kind"], "term": d["term"], "meanings": d["meanings"]}]} for d in slots]
        ref.append({"action": "execute", "request": e.question})
        draft = {"method": "rule" if c["request"] else "none", "needs_review": review,
                 "needs_writing": c["request"] is None}
        return Episode(f"{split}-{kind}-{e.idx}", split, e.db_id, kind, request, glossary, e.query, e.question,
                       sid, slots, None, ref, draft)

    def keep(eps: list[Episode], n: int) -> list[Episode]:
        """Validation/test: keep everything (the team fixes problems). Training: drop weak items."""
        return eps[:n] if review else [ep for ep in eps if not check_episode(ep)][:n]

    episodes = (keep([make_vague(c, "ambiguous") for c in ambiguous_src], target["ambiguous"])
                + keep([make_vague(c, "two_slot") for c in picked_two], target["two_slot"])
                + keep([make_vague(c, "definition") for c in definition_src], target["definition"]))
    for e in clear_src:
        sid = f"{source}-{e.idx}"
        episodes.append(Episode(f"{split}-clear-{e.idx}", split, e.db_id, "clear", e.question,
                                glossary_for(e.db_id, set(), [], sid, []), e.query, e.question, sid, [], None,
                                [{"action": "execute", "request": e.question}],
                                {"method": "original", "needs_review": review, "needs_writing": False}))
    unans = (_trustsql_unanswerable(target["unanswerable"], rng, entries_by_db) if split == "test"
             else _template_unanswerable(split, sorted(dbs), schemas, math.ceil(target["unanswerable"] * spare), rng,
                                         entries_by_db, review))
    episodes += keep(unans, target["unanswerable"])
    stats = {"targets": target, "built": dict(_count(episodes)),
             "available": {"ambiguity_candidates": len(cands),
                           "with_rule_draft": sum(c["request"] is not None for c in cands),
                           "two_slot": len(two), "clear_pool": len(clear_pool)}}
    return episodes, stats


def _count(episodes):
    c = defaultdict(int)
    for ep in episodes:
        c[ep.kind] += 1
    return c


def _template_unanswerable(split, dbs, schemas, n, rng, entries_by_db, review) -> list[Episode]:
    out, i = [], 0
    options = []
    for db in dbs:
        words = {w for t in schemas[db].tables.values() for c in t.columns for w in humanize(c.name).split()}
        words |= {w for t in schemas[db].tables for w in humanize(t).split()}
        words |= {singular(w) for w in words}
        for table, tinfo in schemas[db].tables.items():
            for attr in (PERSON_ATTRIBUTES if is_person_table(tinfo) else THING_ATTRIBUTES):
                if not (set(attr.split()) | {singular(w) for w in attr.split()}) & words:
                    options.append((db, table, attr))
    picked = _round_robin(options, lambda o: o[0], rng, n)
    for db, table, attr in picked:
        entity, entities = singular(table), plural(table)
        request = UNANSWERABLE_TEMPLATES[i % len(UNANSWERABLE_TEMPLATES)].format(attr=attr, entity=entity,
                                                                                 entities=entities)
        i += 1
        explanation = f"The database has no information about the {attr} of {entities}."
        sid = f"template-{db}-{table}-{attr.replace(' ', '_')}"
        glossary = [{k: g[k] for k in ("term", "definition", "aliases")}
                    for g in random.Random(sid).sample(entries_by_db[db], min(MAX_DISTRACTORS, len(entries_by_db[db])))]
        out.append(Episode(f"{split}-unanswerable-{i}", split, db, "unanswerable", request, glossary, None, None, sid, [],
                           {"table": table, "attribute": attr, "explanation": explanation},
                           [{"action": "inspect", "tool": "describe_table", "args": {"table": table}},
                            {"action": "abstain", "missing": explanation}],
                           {"method": "template", "needs_review": review, "needs_writing": False}))
    return out


def _trustsql_unanswerable(n, rng, entries_by_db) -> list[Episode]:
    rows = trustsql.missing_column_questions()
    by_note = defaultdict(list)
    for r in rows:
        by_note[r["note"]].append(r)
    picked = []
    notes = sorted(by_note)
    for k, note in enumerate(notes):
        share = n // len(notes) + (1 if k < n % len(notes) else 0)
        picked += _round_robin(by_note[note], lambda r: r["db_id"], rng, share)
    out = []
    for i, r in enumerate(picked, 1):
        template = r.get("template") or ""
        explanation = f"The database does not store what this question needs ({template})."
        table = template.split(".")[0] if "." in template else None
        glossary = [{k: g[k] for k in ("term", "definition", "aliases")}
                    for g in random.Random(r["id"]).sample(entries_by_db[r["db_id"]],
                                                           min(MAX_DISTRACTORS, len(entries_by_db[r["db_id"]])))]
        ref = ([{"action": "inspect", "tool": "describe_table", "args": {"table": table}}] if table else [])
        ref.append({"action": "abstain", "missing": explanation})
        out.append(Episode(f"test-unanswerable-{i}", "test", r["db_id"], "unanswerable", r["question"], glossary, None,
                           None, f"trustsql-{r['id']}", [],
                           {"table": table, "attribute": template, "explanation": explanation, "note": r["note"]},
                           ref, {"method": "trustsql", "needs_review": True, "needs_writing": False}))
    return out
