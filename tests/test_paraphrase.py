import pytest

from clarifysql.config import data_dir
from clarifysql.episodes.paraphrase import clean, paraphrase_messages
from clarifysql.episodes.schema import EPISODE_DIR

needs_data = pytest.mark.skipif(not (data_dir() / "spider").exists() or not (EPISODE_DIR / "train.jsonl").exists(),
                                reason="needs the episodes and the Spider databases")


def test_clean_strips_labels_and_extra_lines():
    assert clean('Rewritten question: "Which singers are older?"\nThis hides the age.') == "Which singers are older?"


@needs_data
def test_prompt_uses_the_original_question_and_names_the_detail_to_hide():
    from clarifysql.episodes.schema import load_episodes

    ep = next(e for e in load_episodes("train") if e.kind == "ambiguous" and e.slots[0]["kind"] == "threshold")
    user = paraphrase_messages(ep)[1]["content"]
    assert ep.source_question in user and ep.slots[0]["hidden"] in user and ep.request in user


@needs_data
def test_accept_rejects_rewrites_that_leak_or_drift():
    from clarifysql.episodes.paraphrase import accept
    from clarifysql.episodes.schema import load_episodes

    ep = next(e for e in load_episodes("train") if e.kind == "ambiguous" and e.slots[0]["kind"] == "threshold")
    assert accept(ep, ep.request)[0]
    ok, reasons = accept(ep, ep.source_question)  # states the hidden value again
    assert not ok and any("hidden value" in r or "original question" in r for r in reasons)
    ok, reasons = accept(ep, "Show everything?")
    assert not ok and "length very different from the original" in reasons
