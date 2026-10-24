"""The real Hugging Face backend on a tiny random Qwen3 model (CPU): batching and the adapter switch."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
peft = pytest.importorskip("peft")

from clarifysql.config import MODEL_ID  # noqa: E402
from clarifysql.llm.backend import HFBackend  # noqa: E402


@pytest.fixture
def backend():
    from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM

    try:
        tok = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)
    except Exception:
        pytest.skip("Qwen tokenizer is not in the local Hugging Face cache")
    torch.manual_seed(0)
    model = Qwen3ForCausalLM(Qwen3Config(vocab_size=151_936, hidden_size=64, intermediate_size=128,
                                         num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                                         head_dim=16, max_position_embeddings=2048))
    model.generation_config.eos_token_id = tok.eos_token_id
    model.generation_config.pad_token_id = tok.pad_token_id
    b = HFBackend.__new__(HFBackend)  # skip downloading the real 8 GB model
    b.model, b.tok, b.load_seconds, b.has_adapter = model.eval(), tok, 0.0, False
    b.tok.padding_side = "left"
    return b


CHATS = [[{"role": "user", "content": "How many singers are there?"}],
         [{"role": "user", "content": "List the names of all singers from France, sorted by age."}]]


def test_batched_generation_returns_one_result_per_chat_in_order(backend):
    one = [backend.generate([c], max_new_tokens=5)[0].text for c in CHATS]
    both = [g.text for g in backend.generate(CHATS, max_new_tokens=5, batch_size=2)]
    assert both == one  # left padding + greedy decoding: batching does not change the answers
    assert all(g.new_tokens <= 5 for g in backend.generate(CHATS, max_new_tokens=5))


def test_adapter_switch_uses_the_same_weights_with_and_without_lora(backend):
    from peft import LoraConfig, get_peft_model

    with pytest.raises(ValueError):
        backend.generate(CHATS[:1], max_new_tokens=3, adapter=True)  # no adapter loaded yet
    base = backend.generate(CHATS, max_new_tokens=6)
    peft_model = get_peft_model(backend.model, LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"]))
    for name, p in peft_model.named_parameters():
        if "lora_B" in name:
            torch.nn.init.normal_(p, std=2.0)  # make the adapter change the model's behaviour
    backend.model, backend.has_adapter = peft_model.eval(), True
    off = backend.generate(CHATS, max_new_tokens=6, adapter=False)
    on = backend.generate(CHATS, max_new_tokens=6, adapter=True)
    assert [g.text for g in off] == [g.text for g in base]  # adapter off = exactly the original model
    assert [g.text for g in on] != [g.text for g in base]
