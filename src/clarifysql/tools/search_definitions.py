"""Retrieval over a small glossary of definitions with BM25 (the "R" in RAG).

A glossary entry says what a vague phrase means for one database, e.g.
    {"term": "older student", "definition": "a student whose age is above 20"}
If the user's phrase has a stored definition, the agent can use it instead of asking the user.

BM25 scores an entry higher when it shares more (and rarer) words with the query, and does not let
long entries win just because they are long.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

STOPWORDS = {
    "a", "an", "the", "of", "for", "to", "in", "on", "is", "are", "and", "or", "with", "by", "what",
    "which", "who", "show", "list", "find", "me", "all", "that", "whose", "their", "its", "be", "as",
}


def tokenize(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    out = []
    for w in words:
        if w in STOPWORDS:
            continue
        if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
            w = w[:-1]  # crude singular: "students" -> "student", "marks" -> "mark"
        out.append(w)
    return out


@dataclass(frozen=True)
class Definition:
    term: str
    definition: str
    aliases: tuple[str, ...] = ()

    @property
    def text(self) -> str:
        return " ".join((self.term, *self.aliases, self.definition))


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = [tokenize(d) for d in docs]
        self.avg_len = sum(map(len, self.docs)) / max(len(self.docs), 1)
        df = Counter(w for d in self.docs for w in set(d))
        n = len(self.docs)
        self.idf = {w: math.log(1 + (n - c + 0.5) / (c + 0.5)) for w, c in df.items()}

    def scores(self, query: str) -> list[float]:
        q = tokenize(query)
        out = []
        for d in self.docs:
            tf = Counter(d)
            s = 0.0
            for w in q:
                if w not in tf:
                    continue
                f = tf[w]
                s += self.idf[w] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * len(d) / self.avg_len))
            out.append(s)
        return out


def search(glossary: list[Definition], query: str, k: int = 3) -> list[tuple[Definition, float]]:
    """Top-k glossary entries with a positive BM25 score, best first."""
    if not glossary:
        return []
    scores = BM25([g.text for g in glossary]).scores(query)
    ranked = sorted(zip(glossary, scores), key=lambda p: -p[1])
    return [(g, round(s, 3)) for g, s in ranked[:k] if s > 0]


def search_definitions(ctx, query: str, k: int = 3) -> str:
    hits = search(ctx.glossary, query, k)
    if not hits:
        return f'no stored definition matches "{query}"'
    return "\n".join(f"{g.term}: {g.definition}" for g, _ in hits)
