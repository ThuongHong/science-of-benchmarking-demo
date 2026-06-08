"""Systems under test: real Gemma-4 models, plus non-LLM baselines.

A "system" is anything with .name and .predict(item) -> str (raw text). The
evaluator does not care whether the text came from a 4B transformer or from a
coin flip -- that symmetry is the point of having baselines (Measuring what
Matters): a benchmark score only means something if real models clear the dumb
baselines by a wide margin.

  GemmaRunner   : Hugging Face transformers, 4-bit, greedy (deterministic).
  BaselineRunner: random / majority-class / constant answer.
  FakeRunner    : a seeded stand-in so the *pipeline* can be tested on a machine
                  with no GPU. It is NOT used for reported results.

LLM generations are cached to results/predictions/<system>.json keyed by a hash
of (model_id, prompt), so a re-run resumes instantly and is bit-for-bit stable.
"""

from __future__ import annotations

import gc
import hashlib
import json
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "results" / "predictions"
LETTERS = ["A", "B", "C", "D", "E"]


# --------------------------------------------------------------------------- #
# prompt construction
# --------------------------------------------------------------------------- #
def build_prompt(item: dict) -> str:
    if item["benchmark"] == "mmlu":
        opts = "\n".join(f"{LETTERS[i]}. {c}"
                         for i, c in enumerate(item["choices"]))
        return (
            "Answer the multiple-choice question. Reply with the single letter "
            "of the correct option.\n\n"
            f"Question: {item['question']}\n{opts}\n\n"
            "Answer:"
        )
    if item["benchmark"] == "gsm8k":
        return (
            "Solve the math problem. Show brief reasoning, then end with a line "
            "'The answer is <number>'.\n\n"
            f"Problem: {item['question']}\n\nSolution:"
        )
    raise ValueError(item["benchmark"])


# --------------------------------------------------------------------------- #
# prediction cache
# --------------------------------------------------------------------------- #
class _Cache:
    def __init__(self, name: str):
        self.path = CACHE / f"{name}.json"
        self.data: dict[str, str] = {}
        if self.path.exists():
            self.data = json.loads(self.path.read_text())

    @staticmethod
    def key(model_id: str, prompt: str) -> str:
        return hashlib.sha256(f"{model_id}\x00{prompt}".encode()).hexdigest()

    def get(self, k: str) -> str | None:
        return self.data.get(k)

    def put(self, k: str, v: str) -> None:
        self.data[k] = v

    def flush(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, ensure_ascii=False, indent=0))


# --------------------------------------------------------------------------- #
# baselines
# --------------------------------------------------------------------------- #
class BaselineRunner:
    """Non-LLM reference points. kind in {random, majority, constant}."""

    def __init__(self, kind: str):
        self.kind = kind
        self.name = f"baseline-{kind}"

    def predict(self, item: dict) -> str:
        rng = random.Random(f"{self.kind}-{item['id']}")
        if item["benchmark"] == "mmlu":
            n = len(item["choices"])
            if self.kind == "random":
                return LETTERS[rng.randrange(n)]
            if self.kind == "majority":
                return "A"          # "A" is the modal gold label in MMLU
            return "A"              # constant
        # gsm8k: there is no useful constant; emit a plausible small integer
        if self.kind == "random":
            return str(rng.choice([1, 2, 3, 4, 5, 10, 20, 100]))
        return "0"

    def flush(self) -> None:  # symmetry with LLM runner; nothing to persist
        pass

    def unload(self) -> None:
        pass


# --------------------------------------------------------------------------- #
# real model
# --------------------------------------------------------------------------- #
class GemmaRunner:
    """Gemma-4 instruction model via transformers, 4-bit, greedy decoding."""

    def __init__(self, name: str, model_id: str, max_new_tokens: int = 512,
                 load_in_4bit: bool = True):
        import torch
        from transformers import (AutoModelForCausalLM, AutoTokenizer,
                                   BitsAndBytesConfig)

        self.name = name
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.cache = _Cache(name)

        quant = (BitsAndBytesConfig(load_in_4bit=True,
                                    bnb_4bit_compute_dtype=torch.float16)
                 if load_in_4bit else None)
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, device_map="auto", torch_dtype=torch.float16,
            quantization_config=quant)
        self.model.eval()

    def predict(self, item: dict) -> str:
        import torch

        prompt = build_prompt(item)
        ck = self.cache.key(self.model_id, prompt)
        hit = self.cache.get(ck)
        if hit is not None:
            return hit

        messages = [{"role": "user", "content": prompt}]
        # return_dict=True gives a BatchEncoding (input_ids + attention_mask);
        # newer transformers no longer return a bare tensor here.
        enc = self.tok.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt",
            return_dict=True).to(self.model.device)
        input_len = enc["input_ids"].shape[1]
        with torch.no_grad():
            out = self.model.generate(
                **enc, max_new_tokens=self.max_new_tokens,
                do_sample=False,           # greedy -> deterministic
                pad_token_id=self.tok.eos_token_id)
        text = self.tok.decode(out[0][input_len:],
                               skip_special_tokens=True).strip()
        self.cache.put(ck, text)
        return text

    def flush(self) -> None:
        self.cache.flush()

    def unload(self) -> None:
        """Free GPU memory so the next model can load on its own."""
        import torch

        self.flush()
        self.model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# --------------------------------------------------------------------------- #
# fake model (no-GPU pipeline test only)
# --------------------------------------------------------------------------- #
class FakeRunner:
    """Seeded stand-in. `skill` in [0,1] controls how often it is correct.

    Crafts answers with messy surface form (trailing '.', '$', extra prose) so
    the metric-brittleness and letter-extraction paths actually get exercised.
    """

    def __init__(self, name: str, skill: float = 0.8):
        self.name = name
        self.skill = skill

    def predict(self, item: dict) -> str:
        rng = random.Random(f"{self.name}-{item['id']}")
        correct = rng.random() < self.skill
        if item["benchmark"] == "mmlu":
            letter = (item["answer"] if correct
                      else rng.choice([l for l in LETTERS[:len(item["choices"])]
                                       if l != item["answer"]]))
            return f"The answer is ({letter})."
        # gsm8k: phrase final answer with currency/punct so naive grader fails
        gold = item["answer"]
        val = gold if correct else str(int(float(gold)) + rng.randint(1, 9))
        return f"Working through it... The answer is ${val}."

    def flush(self) -> None:
        pass

    def unload(self) -> None:
        pass


# --------------------------------------------------------------------------- #
# factory
# --------------------------------------------------------------------------- #
def make_system(s: dict):
    """Build ONE system from a spec entry: {kind: gemma|baseline|fake, ...}.

    Built lazily, one at a time, so only a single Gemma model sits on the GPU at
    once -- loading every model up front would exhaust VRAM and force a CPU
    offload that 4-bit bitsandbytes refuses.
    """
    kind = s["kind"]
    if kind == "gemma":
        return GemmaRunner(
            s["name"], s["model_id"],
            max_new_tokens=s.get("max_new_tokens", 512),
            load_in_4bit=s.get("load_in_4bit", True))
    if kind == "baseline":
        return BaselineRunner(s["baseline"])
    if kind == "fake":
        return FakeRunner(s["name"], skill=s.get("skill", 0.8))
    raise ValueError(f"unknown system kind {kind!r}")


def build_systems(spec: list[dict]):
    """Eagerly build every system (kept for callers that want them all)."""
    return [make_system(s) for s in spec]
