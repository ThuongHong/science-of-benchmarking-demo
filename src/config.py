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
# `params_b` (billions of params) is metadata for reporting only -- it lets the
# analysis show the weight class so a small model's low score isn't mistaken for
# a family effect. E2B/E4B are the same-family scaling pair; the ~3-4B models
# form the weight-matched cross-family panel.
GEMMA_SYSTEMS = [
    {"kind": "gemma", "name": "gemma-4-e2b", "model_id": "google/gemma-4-E2B-it",
     "params_b": 2.3},
    {"kind": "gemma", "name": "gemma-4-e4b", "model_id": "google/gemma-4-E4B-it",
     "params_b": 4.5},
]

# A stronger, cross-family rung that still fits a single T4 in 4-bit (~3GB).
# Qwen3.5 is a reasoning model: turn thinking off so it answers directly, and
# give it extra tokens for math chains of thought.
QWEN_STRONG = {
    "kind": "hf", "name": "qwen3.5-4b", "model_id": "Qwen/Qwen3.5-4B",
    "max_new_tokens": 768, "chat_template_kwargs": {"enable_thinking": False},
    "params_b": 4.0,
}

# Same-size (~3-4B) peers from other families. All ungated, non-thinking, fit a
# single T4 in 4-bit. Purpose: at a fixed size, does the benchmark ranking agree
# across MMLU and GSM8K? Disagreement = "the benchmark picks the winner".
PEERS = [
    {"kind": "hf", "name": "phi-3.5-mini",
     "model_id": "microsoft/Phi-3.5-mini-instruct", "params_b": 3.8},
    {"kind": "hf", "name": "qwen2.5-3b",
     "model_id": "Qwen/Qwen2.5-3B-Instruct", "params_b": 3.1},
]

# A *domain specialist*, deliberately off the weight-matched panel (7B, math-
# tuned). On the general panel the MMLU and GSM8K rankings agree perfectly
# (Spearman rho = 1.0). A math specialist should climb on GSM8K while staying
# mediocre on MMLU -- if so, it breaks the tie (rho < 1) and turns the
# rank-stability section into a positive result: the benchmark, not raw skill,
# picks the winner. Needs room for a chain-of-thought, so give it more tokens.
MATH_SPECIALIST = {
    "kind": "hf", "name": "qwen2.5-math-7b",
    "model_id": "Qwen/Qwen2.5-Math-7B-Instruct",
    "max_new_tokens": 768, "params_b": 7.0,
}

BASELINES = [
    {"kind": "baseline", "baseline": "random"},
    {"kind": "baseline", "baseline": "majority"},
]

# real run = Gemma rungs (E2B, E4B) + Qwen3.5-4B + ~4B peers + math specialist
# + baselines
SYSTEMS = GEMMA_SYSTEMS + [QWEN_STRONG] + PEERS + [MATH_SPECIALIST] + BASELINES

# name -> size (billions), for reporting the weight class in the analysis
PARAMS_B = {s["name"]: s["params_b"]
            for s in GEMMA_SYSTEMS + [QWEN_STRONG] + PEERS + [MATH_SPECIALIST]}

# no-GPU pipeline test: two fake rungs of different skill + baselines
FAKE_SYSTEMS = [
    {"kind": "fake", "name": "fake-small", "skill": 0.55},
    {"kind": "fake", "name": "fake-large", "skill": 0.85},
] + BASELINES
