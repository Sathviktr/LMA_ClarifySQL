"""The whole training loop on a tiny random Qwen3 model (CPU): what runs on the T4, at toy size."""

import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")

from clarifysql.config import MODEL_ID  # noqa: E402
from clarifysql.train import qlora  # noqa: E402

ACTIONS = ['{"action":"clarify","slots":[{"name":"threshold","term":"older singers","meanings":["age above some '
           'value (which?)"]}],"question":"What age counts as \\"older\\"?"}',
           '{"action":"execute","request":"List the names of singers from France."}',
           '{"action":"inspect","tool":"search_definitions","args":{"query":"older singers"}}',
           '{"action":"abstain","missing":"The database has no email addresses of singers."}']
REQUESTS = ["Show the older singers.", "List the names of singers from France.", "Who are the older singers?",
            "What is the email of each singer?"]


def _states(n: int) -> list[dict]:
    """Short states in the real format (the real ones are ~450 tokens: too much memory for a laptop test)."""
    return [{"id": f"s{i}", "messages": [{"role": "system", "content": "Answer with one JSON action."},
                                         {"role": "user", "content": f"Request: {REQUESTS[i % 4]}\nNext action:"}],
             "target": ACTIONS[i % 4], "target_action": json.loads(ACTIONS[i % 4])["action"]} for i in range(n)]


def _tiny_model(tok):
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(0)
    model = Qwen3ForCausalLM(Qwen3Config(vocab_size=151_936, hidden_size=64, intermediate_size=128,
                                         num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                                         head_dim=16, max_position_embeddings=2048))
    model.generation_config.eos_token_id = tok.eos_token_id
    model.generation_config.pad_token_id = tok.pad_token_id
    return model


@pytest.fixture
def tok():
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)
    except Exception:
        pytest.skip("Qwen tokenizer is not in the local Hugging Face cache")


def test_training_learns_saves_every_epoch_and_the_saved_adapter_reloads(tok, tmp_path, monkeypatch):
    from peft import PeftModel

    from clarifysql.llm.backend import HFBackend

    monkeypatch.setattr(qlora, "load_states", lambda name: _states(8 if name.startswith("train") else 4))
    monkeypatch.setattr(qlora, "POLICY_MAX_TOKENS", 12)
    cfg = qlora.TrainConfig(name="tiny", epochs=3, lr=1e-2, grad_accum=3, r=8, alpha=16, eval_batch_size=4)
    model = _tiny_model(tok)
    base_before = {k: v.clone() for k, v in model.state_dict().items()}
    log_path = tmp_path / "results" / "tiny.json"
    log = qlora.train(cfg, model, tok, tmp_path / "adapters", log_path)

    # 8 states, 3 per step -> 3 optimizer steps per epoch (the last one with 2 states)
    assert log["optimizer_steps"] == 9 and len(log["steps"]) == 9 and log["train_states"] == 8
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    assert trainable and all("lora_" in n for n in trainable)        # only the adapter trains
    base_after = model.state_dict()  # get_peft_model put the LoRA layers into this same model
    frozen = {k: k.replace("base_layer.", "") for k in base_after if "lora_" not in k}
    assert set(frozen.values()) == set(base_before)
    assert all(torch.equal(base_after[k], base_before[old]) for k, old in frozen.items())  # base never changes
    first, last = log["epochs"][0], log["epochs"][-1]
    assert last["train_loss"] < first["train_loss"]                  # it learns the 8 targets
    assert all(e["validation"]["states"] == 4 for e in log["epochs"]) and log["best_epoch"] in (1, 2, 3)
    assert json.loads(log_path.read_text())["epochs"][-1]["epoch"] == 3
    assert (tmp_path / "results" / "tiny_epoch-3_answers.jsonl").exists()
    for k in (1, 2, 3):
        assert (tmp_path / "adapters" / f"epoch-{k}" / "adapter_config.json").exists()

    # the saved adapter, loaded the way scripts/run_agent.py --adapter loads it, writes the same actions
    chats = [s["messages"] for s in _states(4)]
    trained = HFBackend.wrap(model.eval(), tok, has_adapter=True).generate(chats, max_new_tokens=8, adapter=True)
    reloaded = PeftModel.from_pretrained(_tiny_model(tok), str(tmp_path / "adapters" / "epoch-3")).eval()
    again = HFBackend.wrap(reloaded, tok, has_adapter=True).generate(chats, max_new_tokens=8, adapter=True)
    assert [g.text for g in again] == [g.text for g in trained]
