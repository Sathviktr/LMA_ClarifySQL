"""CPU smoke tests of the pilot code with a tiny, randomly initialised Qwen3 model (no download needed)."""

import math

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytest.importorskip("peft")

from clarifysql import pilot  # noqa: E402
from clarifysql.config import MODEL_ID  # noqa: E402

TINY = dict(vocab_size=1000, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=2048)


def tiny_model(**overrides):
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(0)
    return Qwen3ForCausalLM(Qwen3Config(**{**TINY, **overrides}))


def qwen_tokenizer():
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)
    except Exception:
        pytest.skip("Qwen tokenizer is not in the local Hugging Face cache")


def test_formula_matches_real_peft_count_and_training_runs():
    results, peft_model = pilot.measure_qlora_steps(tiny_model(), seq_lens=(16, 32), steps=2)
    assert results["trainable_params"] == pilot.lora_param_count(TINY, r=16)
    for run in results["runs"]:
        assert run["status"] == "ok"
        assert math.isfinite(run["loss_last"])
    names = {n for n, p in peft_model.named_parameters() if p.requires_grad}
    assert names and all("lora_" in n for n in names)  # only adapter weights are trained


def test_generation_sanity_and_latency_on_cpu():
    tok = qwen_tokenizer()
    model = tiny_model(vocab_size=151_936)
    model.generation_config.eos_token_id = tok.eos_token_id
    model.generation_config.pad_token_id = tok.pad_token_id
    sanity = pilot.fp16_sanity(model, tok, max_new_tokens=4)
    assert sanity["logits_finite"] is True
    gen = pilot.measure_generation(model, tok, batch_sizes=(1, 2), max_new_tokens=5)
    assert [g["new_tokens"] for g in gen] == [5, 10]
    lat = pilot.measure_call_latency(model, tok, prompt_tokens=100, new_tokens=3)
    assert lat["prompt_tokens"] > 100 and lat["seconds"] > 0
