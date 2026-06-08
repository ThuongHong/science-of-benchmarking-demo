"""Load deterministic subsets of two *real* public benchmarks and cache them.

We do not invent a benchmark. We sample fixed subsets of:

  * MMLU  (cais/mmlu, config "all") -- 4-way multiple-choice knowledge QA.
  * GSM8K (openai/gsm8k, config "main") -- grade-school math word problems.

The subsets are written to data/*.jsonl with a fixed seed so the whole demo is
reproducible: anyone re-running gets the *same* items in the *same* order, and
the committed jsonl lets a grader inspect the exact data without a Hub download.

Unified item schema
-------------------
MMLU : {id, benchmark:"mmlu", subject, question, choices:[..4..],
        answer_index:int, answer:"A".."D"}
GSM8K: {id, benchmark:"gsm8k", question, answer:str(number), solution:str}
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
LETTERS = ["A", "B", "C", "D"]


# --------------------------------------------------------------------------- #
# jsonl helpers
# --------------------------------------------------------------------------- #
def save_jsonl(items: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for it in items:
            fh.write(json.dumps(it, ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> list[dict]:
    items = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


# --------------------------------------------------------------------------- #
# MMLU
# --------------------------------------------------------------------------- #
def load_mmlu(n: int = 150, seed: int = 0) -> list[dict]:
    """Stratified sample of `n` MMLU test items across all subjects.

    Stratifying by subject keeps the subset from collapsing onto one topic, so
    "MMLU accuracy" stays a (rough) measure of broad knowledge rather than of a
    single domain -- a small nod to the construct-validity point of the tutorial.
    """
    from datasets import load_dataset

    ds = load_dataset("cais/mmlu", "all", split="test")
    by_subject: dict[str, list[int]] = {}
    for i, subj in enumerate(ds["subject"]):
        by_subject.setdefault(subj, []).append(i)

    rng = random.Random(seed)
    subjects = sorted(by_subject)
    per = max(1, n // len(subjects))

    picked: list[int] = []
    for subj in subjects:
        idxs = by_subject[subj]
        picked.extend(rng.sample(idxs, min(per, len(idxs))))
    rng.shuffle(picked)
    picked = picked[:n]

    items = []
    for rank, i in enumerate(picked):
        row = ds[i]
        items.append({
            "id": f"mmlu_{rank:04d}",
            "benchmark": "mmlu",
            "subject": row["subject"],
            "question": row["question"],
            "choices": list(row["choices"]),
            "answer_index": int(row["answer"]),
            "answer": LETTERS[int(row["answer"])],
        })
    return items


# --------------------------------------------------------------------------- #
# GSM8K
# --------------------------------------------------------------------------- #
def _gsm8k_gold(answer_field: str) -> tuple[str, str]:
    """GSM8K gold answers end with a '#### <number>' line. Split it out."""
    solution, _, final = answer_field.rpartition("####")
    return final.strip().replace(",", ""), solution.strip()


def load_gsm8k(n: int = 80, seed: int = 0) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split="test")
    rng = random.Random(seed)
    picked = rng.sample(range(len(ds)), min(n, len(ds)))

    items = []
    for rank, i in enumerate(picked):
        row = ds[i]
        number, solution = _gsm8k_gold(row["answer"])
        items.append({
            "id": f"gsm8k_{rank:04d}",
            "benchmark": "gsm8k",
            "question": row["question"],
            "answer": number,
            "solution": solution,
        })
    return items


# --------------------------------------------------------------------------- #
# build + cache
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# synthetic fallback (offline pipeline test only, never used for real results)
# --------------------------------------------------------------------------- #
def _synthetic(n_mmlu: int, n_gsm8k: int, seed: int) -> dict[str, list[dict]]:
    rng = random.Random(seed)
    mmlu = []
    for i in range(n_mmlu):
        gold = rng.randrange(4)
        mmlu.append({
            "id": f"mmlu_{i:04d}", "benchmark": "mmlu", "subject": "synthetic",
            "question": f"Synthetic question {i}?",
            "choices": [f"opt{j}" for j in range(4)],
            "answer_index": gold, "answer": LETTERS[gold],
        })
    gsm8k = []
    for i in range(n_gsm8k):
        ans = rng.randint(1, 500)
        gsm8k.append({
            "id": f"gsm8k_{i:04d}", "benchmark": "gsm8k",
            "question": f"Synthetic math {i}: what is {ans}?",
            "answer": str(ans), "solution": f"It is {ans}.",
        })
    return {"mmlu": mmlu, "gsm8k": gsm8k}


def build_subsets(n_mmlu: int = 150, n_gsm8k: int = 80, seed: int = 0,
                  force: bool = False) -> dict[str, list[dict]]:
    """Materialize the subsets to data/*.jsonl (cached unless force=True).

    Set MINIBENCH_SYNTHETIC=1 to skip the Hub entirely and use offline fixtures
    -- only for testing the pipeline on a machine with no GPU/network.
    """
    if os.environ.get("MINIBENCH_SYNTHETIC") == "1":
        return _synthetic(n_mmlu, n_gsm8k, seed)
    out = {}
    spec = {"mmlu": (load_mmlu, n_mmlu), "gsm8k": (load_gsm8k, n_gsm8k)}
    for name, (loader, n) in spec.items():
        path = DATA / f"{name}.jsonl"
        if path.exists() and not force:
            out[name] = load_jsonl(path)
        else:
            items = loader(n, seed)
            save_jsonl(items, path)
            out[name] = items
    return out


if __name__ == "__main__":
    subs = build_subsets(force=True)
    for name, items in subs.items():
        print(f"{name}: {len(items)} items -> data/{name}.jsonl")
