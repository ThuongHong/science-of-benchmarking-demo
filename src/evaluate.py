"""Run every system over every benchmark item and write per-item results.

Output: results/per_item.csv with one row per (system, item) and columns:
  system, benchmark, id, orig_id, perturbation, subject, gold,
  prediction, extracted, correct, correct_naive

`correct`       -- primary metric (MMLU letter match / GSM8K robust numeric).
`correct_naive` -- GSM8K only: the brittle last-line grader (== correct for MMLU).
The two columns drive the "metric != construct" analysis downstream.

Usage:
  python src/evaluate.py            # real run (needs GPU + gated Gemma access)
  python src/evaluate.py --fake     # no-GPU pipeline smoke test
"""

from __future__ import annotations

import argparse

import pandas as pd

import config
from data import build_subsets
from metrics import (extract_letter, extract_number_robust, mmlu_score,
                     gsm8k_score_naive, gsm8k_score_robust)
from models import make_system
from perturb import build_perturbations
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _row(system_name: str, item: dict, prediction: str) -> dict:
    bench = item["benchmark"]
    if bench == "mmlu":
        extracted = extract_letter(prediction)
        correct = mmlu_score(item, prediction)
        correct_naive = correct
    else:
        val = extract_number_robust(prediction)
        extracted = "" if val is None else (
            str(int(val)) if val == int(val) else str(val))
        correct = gsm8k_score_robust(item, prediction)
        correct_naive = gsm8k_score_naive(item, prediction)
    return {
        "system": system_name,
        "benchmark": bench,
        "id": item["id"],
        "orig_id": item.get("orig_id", ""),
        "perturbation": item.get("perturbation", ""),
        "subject": item.get("subject", ""),
        "gold": item["answer"],
        "prediction": prediction.replace("\n", " ⏎ ")[:300],
        "extracted": extracted or "",
        "correct": int(correct),
        "correct_naive": int(correct_naive),
    }


def run(system_spec: list[dict]) -> pd.DataFrame:
    subs = build_subsets(config.N_MMLU, config.N_GSM8K, config.SEED)
    perturbed = build_perturbations(subs["mmlu"], config.N_PERTURB, config.SEED)
    # tag the natural sets so we can separate "main" from "perturbation"
    items = (subs["mmlu"] + subs["gsm8k"] + perturbed)

    rows = []
    # Build one system at a time and free it before the next, so only a single
    # Gemma model occupies the GPU at any moment (avoids VRAM exhaustion).
    for spec in system_spec:
        sys = make_system(spec)
        print(f"[{sys.name}] {len(items)} items...")
        for k, item in enumerate(items):
            rows.append(_row(sys.name, item, sys.predict(item)))
            if (k + 1) % 50 == 0:
                sys.flush()
                print(f"  {sys.name}: {k + 1}/{len(items)}")
        sys.unload()
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fake", action="store_true",
                    help="use seeded fake models (no GPU needed)")
    args = ap.parse_args()

    spec = config.FAKE_SYSTEMS if args.fake else config.SYSTEMS
    df = run(spec)

    out = ROOT / "results" / "per_item.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nWrote {out}  ({len(df)} rows)")


if __name__ == "__main__":
    main()
