"""The controller, the simulated user and the scoring, with scripted stand-ins for the model (no GPU)."""

import pytest

from clarifysql.agent.controller import Controller, trace, user_values
from clarifysql.config import Budgets, data_dir
from clarifysql.episodes.schema import EPISODE_DIR, Episode
from clarifysql.eval.metrics import macro_f1, names_missing
from clarifysql.llm.backend import MockBackend
from clarifysql.simuser.simulated_user import NOT_SURE, SimulatedUser, choice_messages, parse_choices

needs_data = pytest.mark.skipif(not (data_dir() / "spider").exists() or not (EPISODE_DIR / "validation.jsonl").exists(),
                                reason="needs the episodes and the Spider databases")


def test_macro_f1_by_hand():
    gold = ["clarify", "execute", "execute", "abstain"]
    pred = ["clarify", "clarify", "execute", "none"]
    f1, per = macro_f1(gold, pred)
    # clarify: P=1/2 R=1 F1=0.667; execute: P=1 R=1/2 F1=0.667; abstain: P=0 R=0 F1=0
    assert per["clarify"]["f1"] == pytest.approx(0.667, abs=1e-3) and per["abstain"]["f1"] == 0.0
    assert f1 == pytest.approx((0.667 + 0.667 + 0) / 3, abs=1e-3)


def test_user_values_reads_numbers_from_answers():
    assert user_values(["I mean more than 4.", "I mean between 1935 and 1939."]) == {"4", "1935", "1939"}


def test_simulated_user_reads_only_the_question_and_answers_one_slot():
    slots = [{"kind": "threshold", "term": "many customers", "hidden": "more than 4", "value": "> 4",
              "answer": "I mean more than 4."},
             {"kind": "ranking", "term": "city ranks highest", "hidden": "", "value": "population",
              "answer": "Rank them by population."}]
    chat = choice_messages("Show it.", "How many customers count as many?", slots)
    assert "slots" not in chat[1]["content"] and "C) none of these" in chat[1]["content"]
    assert parse_choices("B", 2) == [1] and parse_choices("C", 2) == [] and parse_choices("?", 2) == []
    assert parse_choices("B, A", 2) == [1, 0] and parse_choices("A,C", 2) == [0]
    user = SimulatedUser(MockBackend(lambda c, a: "A"))
    items = [{"episode": "e", "request": "Show it.", "question": "How many?", "slots": slots},
             {"episode": "f", "request": "Clear.", "question": "Really?", "slots": []}]
    replies = user.answer_batch(items)
    assert replies[0].text == "I mean more than 4." and replies[0].slot_indices == [0]
    assert replies[1].slot_indices == [] and len(user.log) == 2
    assert SimulatedUser(MockBackend(lambda c, a: "C")).answer_batch(items[:1])[0].text == NOT_SURE


def test_one_question_about_two_details_gets_both_answers_each_named():
    slots = [{"kind": "threshold", "term": "long length", "hidden": "longer than 100", "value": "> 100",
              "answer": "I mean longer than 100."},
             {"kind": "threshold", "term": "high replacement cost", "hidden": "more than 200", "value": "> 200",
              "answer": "I mean more than 200."}]
    item = {"episode": "e", "request": "Long films except costly ones.", "question": "How long, and how costly?",
            "slots": slots}
    reply = SimulatedUser(MockBackend(lambda c, a: "A,B")).answer_batch([item])[0]
    assert reply.slot_indices == [0, 1] and reply.choice == "A,B"
    assert reply.text == ('For "long length": I mean longer than 100. '
                          'For "high replacement cost": I mean more than 200.')


def test_names_missing():
    ep = Episode("u", "test", "db", "unanswerable", "x", missing={"attribute": "email address", "explanation": ""},
                 draft={"method": "template"})
    assert names_missing("There are no email addresses of students stored.", ep)
    assert not names_missing("I cannot answer that.", ep)
    ts = Episode("t", "test", "db", "unanswerable", "x", missing={"attribute": "Highschooler.Capacity"},
                 draft={"method": "trustsql"})
    assert names_missing("No capacity is recorded for high schoolers.", ts)


@needs_data
def test_reference_policy_scores_perfectly():
    from clarifysql.agent.mocks import GoldWriter, KeywordUser, ReferencePolicy
    from clarifysql.episodes.schema import load_episodes
    from clarifysql.eval.metrics import score_trace, summarize

    episodes = [ep for ep in load_episodes("validation") if not ep.draft.get("needs_writing")]
    runs = Controller(ReferencePolicy(episodes), GoldWriter(episodes), SimulatedUser(KeywordUser())).run(episodes)
    s = summarize([score_trace(trace(r), r.ep) for r in runs])
    assert s["execution_accuracy"] == 1.0 and s["decision_macro_f1"] == 1.0 and s["correct_abstention_rate"] == 1.0
    assert s["unnecessary_question_rate_clear"] == 0.0 and s["limit_rate"] == 0.0


@needs_data
def test_budgets_format_retry_and_repairs():
    from clarifysql.episodes.schema import load_episodes

    ep = next(e for e in load_episodes("validation") if e.kind == "ambiguous" and not e.draft.get("needs_writing"))

    # a policy that always asks: the third question exceeds the budget of 2, so the episode ends as "limit"
    asker = MockBackend(lambda c, a: '{"action":"clarify","question":"Which one?"}')
    r = Controller(asker, None, SimulatedUser(MockBackend(lambda c, a: "Z"))).run([ep])[0]
    assert r.outcome == "limit" and r.questions == 2 and r.decision == "clarify"

    # an invalid answer gets exactly one retry, then the episode ends as "invalid"
    r = Controller(MockBackend(lambda c, a: "I think we should ask."), None, None).run([ep])[0]
    assert r.outcome == "invalid" and r.invalid_outputs == 2 and r.format_retries == 1

    # SQL errors are repaired at most twice, within the 8-call budget
    execute = MockBackend(lambda c, a: '{"action":"execute","request":"Show everything."}')
    broken = MockBackend(lambda c, a: "```sql\nSELECT nothing FROM nowhere\n```")
    r = Controller(execute, broken, None).run([ep])[0]
    assert r.outcome == "executed" and r.repairs == 2 and r.sql_status == "sql_error" and r.calls == 4
    r = Controller(execute, broken, None, budgets=Budgets(max_model_calls=2)).run([ep])[0]
    assert r.outcome == "limit" and r.calls == 2


@needs_data
def test_a_detail_asked_again_gets_the_same_answer_again():
    from clarifysql.episodes.schema import load_episodes

    ep = next(e for e in load_episodes("validation") if e.kind == "ambiguous" and not e.draft.get("needs_writing"))
    asker = MockBackend(lambda c, a: '{"action":"clarify","question":"Which one?"}')
    r = Controller(asker, None, SimulatedUser(MockBackend(lambda c, a: "A"))).run([ep])[0]
    assert [e["reply"] for e in r.events if e["role"] == "user"] == [ep.slots[0]["answer"]] * 2


@needs_data
def test_intent_check_asks_for_a_repair_when_the_users_value_is_missing():
    from clarifysql.episodes.schema import load_episodes

    ep = next(e for e in load_episodes("validation") if e.kind == "ambiguous" and e.slots[0]["kind"] == "threshold"
              and not e.draft.get("needs_writing"))
    steps = iter(['{"action":"clarify","question":"What cut-off?"}', '{"action":"execute","request":"Do it."}'])
    policy = MockBackend(lambda c, a: next(steps))
    sqls = iter(["SELECT 1", ep.gold_sql])
    writer = MockBackend(lambda c, a: next(sqls))
    r = Controller(policy, writer, SimulatedUser(MockBackend(lambda c, a: "A"))).run([ep])[0]
    writer_events = [e for e in r.events if e["role"] == "writer"]
    assert r.repairs == 1 and writer_events[0]["intent_missing"] and not writer_events[1]["intent_missing"]
    assert r.sql == ep.gold_sql and r.outcome == "executed"
