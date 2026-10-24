"""Scripted stand-ins for the three model roles, for laptop tests and the notebooks' pipeline checks.

ReferencePolicy  follows each episode's reference path exactly (an upper bound for the controller)
KeywordUser      picks the multiple-choice option whose glossary term shares the most words with the question
GoldWriter       answers with the gold SQL when it receives an episode's fully specified question
"""

from __future__ import annotations

import re

from clarifysql.agent.actions import format_action
from clarifysql.episodes.schema import Episode
from clarifysql.llm.backend import Generation

_STEP = re.compile(r"^\d+\. ", re.M)


def _request_of(chat: list[dict]) -> str:
    m = re.search(r"^Request: (.*)$", chat[-1]["content"], re.M)
    return m.group(1).strip() if m else ""


class _Mock:
    def __init__(self):
        self.calls = 0

    def _gen(self, text: str) -> Generation:
        self.calls += 1
        return Generation(text, 10, len(text.split()), 0.0)


class ReferencePolicy(_Mock):
    def __init__(self, episodes: list[Episode], with_slots: bool = True):
        super().__init__()
        self.by_request = {ep.request: ep for ep in episodes}
        self.with_slots = with_slots

    def generate(self, chats, max_new_tokens=192, adapter=True, batch_size=16):
        out = []
        for chat in chats:
            ep = self.by_request[_request_of(chat)]
            k = len(_STEP.findall(chat[-1]["content"].split("Steps so far:")[-1])) if "Steps so far:" in \
                chat[-1]["content"] else 0
            out.append(self._gen(format_action(ep.reference[min(k, len(ep.reference) - 1)], self.with_slots)))
        return out


class KeywordUser(_Mock):
    def generate(self, chats, max_new_tokens=4, adapter=False, batch_size=16):
        out = []
        for chat in chats:
            text = chat[-1]["content"]
            question = re.search(r'clarifying question: "(.*)"\n', text).group(1).lower()
            options = re.findall(r'^([A-H])\) (.*)$', text, re.M)
            best, score = options[-1][0], 0
            for letter, desc in options[:-1]:
                term = re.search(r'"([^"]+)"', desc)
                words = set(term.group(1).lower().split()) if term else set()
                s = sum(w in question for w in words)
                if s > score:
                    best, score = letter, s
            out.append(self._gen(best))
        return out


class GoldWriter(_Mock):
    def __init__(self, episodes: list[Episode]):
        super().__init__()
        self.gold = {}
        for ep in episodes:
            if ep.gold_sql:
                self.gold[ep.source_question] = ep.gold_sql
                self.gold[ep.request] = ep.gold_sql if ep.kind == "clear" else self.gold.get(ep.request, "SELECT 1")

    def generate(self, chats, max_new_tokens=256, adapter=False, batch_size=16):
        out = []
        for chat in chats:
            request = chat[1]["content"].split("Request: ")[-1].strip()
            out.append(self._gen(f"```sql\n{self.gold.get(request, 'SELECT 1')}\n```"))
        return out
