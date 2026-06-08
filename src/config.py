"""Central knobs for the demo: subset sizes and the systems under test.

Edit SYSTEMS to add/remove models. The two Gemma-4 sizes below both fit a free
Colab T4 (E2B ~1.5GB, E4B ~5GB in 4-bit). Add a third rung (e.g. the 26B-A4B
MoE) only on Kaggle/A100 -- it will OOM a single T4.
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

BASELINES = [
    {"kind": "baseline", "baseline": "random"},
    {"kind": "baseline", "baseline": "majority"},
]

# real run = Gemma rungs + baselines
SYSTEMS = GEMMA_SYSTEMS + BASELINES

# no-GPU pipeline test: two fake rungs of different skill + baselines
FAKE_SYSTEMS = [
    {"kind": "fake", "name": "fake-small", "skill": 0.55},
    {"kind": "fake", "name": "fake-large", "skill": 0.85},
] + BASELINES
