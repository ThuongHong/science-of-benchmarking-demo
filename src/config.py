"""Central knobs for the demo: subset sizes and the systems under test.

Edit SYSTEMS to add/remove models. The default three rungs all fit a single
free T4 in 4-bit: Gemma-4 E2B (~1.5GB), Gemma-4 E4B (~5GB) and Qwen3.5-4B
(~3GB, a stronger cross-family reference).
"""

# subset sizes (kept modest so a free T4 finishes in well under an hour)
N_MMLU = 150
N_GSM8K = 80
N_PERTURB = 50
SEED = 0

# Gemma models are gated on the Hub: accept the license once and pass an
# HF token (notebook does huggingface_hub.login) before the first run.
GEMMA_SYSTEMS = [
    {"kind": "gemma", "name": "gemma-4-e2b", "model_id": "google/gemma-4-E2B-it"},
    {"kind": "gemma", "name": "gemma-4-e4b", "model_id": "google/gemma-4-E4B-it"},
]

# A stronger, cross-family rung that still fits a single T4 in 4-bit (~3GB).
# Qwen3.5 is a reasoning model: turn thinking off so it answers directly, and
# give it extra tokens for math chains of thought.
QWEN_STRONG = {
    "kind": "hf", "name": "qwen3.5-4b", "model_id": "Qwen/Qwen3.5-4B",
    "max_new_tokens": 768, "chat_template_kwargs": {"enable_thinking": False},
}

# Same-size (~3-4B) peers from other families. All ungated, non-thinking, fit a
# single T4 in 4-bit. Purpose: at a fixed size, does the benchmark ranking agree
# across MMLU and GSM8K? Disagreement = "the benchmark picks the winner".
PEERS = [
    {"kind": "hf", "name": "phi-3.5-mini",
     "model_id": "microsoft/Phi-3.5-mini-instruct"},
    {"kind": "hf", "name": "qwen2.5-3b",
     "model_id": "Qwen/Qwen2.5-3B-Instruct"},
]

BASELINES = [
    {"kind": "baseline", "baseline": "random"},
    {"kind": "baseline", "baseline": "majority"},
]

# real run = Gemma rungs (E2B, E4B) + Qwen3.5-4B + ~4B peers + baselines
SYSTEMS = GEMMA_SYSTEMS + [QWEN_STRONG] + PEERS + BASELINES

# no-GPU pipeline test: two fake rungs of different skill + baselines
FAKE_SYSTEMS = [
    {"kind": "fake", "name": "fake-small", "skill": 0.55},
    {"kind": "fake", "name": "fake-large", "skill": 0.85},
] + BASELINES
