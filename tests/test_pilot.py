import json

from clarifysql import pilot

QWEN3_4B = {
    "hidden_size": 2560, "intermediate_size": 9728, "num_hidden_layers": 36,
    "num_attention_heads": 32, "num_key_value_heads": 8, "head_dim": 128,
}


def test_lora_count_for_qwen3_4b_matches_proposal():
    n = pilot.lora_param_count(QWEN3_4B, r=16)
    assert n == 33_030_144
    assert 32e6 < n < 34e6  # proposal Section 9: "about 33M trainable parameters"


def test_lora_count_is_linear_in_rank():
    assert 2 * pilot.lora_param_count(QWEN3_4B, r=8) == pilot.lora_param_count(QWEN3_4B, r=16)


def test_lora_count_per_module():
    # q_proj maps 2560 -> 32 heads * 128 = 4096, so one LoRA pair adds 16 * (2560 + 4096) numbers per layer
    assert pilot.lora_param_count(QWEN3_4B, r=16, targets=("q_proj",)) == 36 * 16 * (2560 + 4096)


def test_summarize_handles_partial_results():
    text = pilot.summarize({"qlora": {"trainable_params": 10,
                                      "runs": [{"seq_len": 8, "status": "out_of_memory"}]}})
    assert "out_of_memory" in text and "10" in text


def test_save_results_roundtrip(tmp_path):
    path = pilot.save_results({"a": 1, "b": [1, 2]}, tmp_path / "x" / "pilot.json")
    assert json.loads(path.read_text(encoding="utf-8")) == {"a": 1, "b": [1, 2]}
