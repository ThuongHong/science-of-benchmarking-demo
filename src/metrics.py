"""Scoring metrics, written to *expose* how brittle benchmark scoring can be.

Two benchmarks, two scoring concerns the tutorial cares about:

  * MMLU  -- extracting the chosen letter from free-form model text. A grader
    that only accepts a bare "A" silently misreads "The answer is A." and tanks
    the score. We extract robustly and keep a flag for "letter was present but
    not at position 0", to quantify how much a naive grader would have lost.

  * GSM8K -- numeric answer extraction. We implement TWO graders on purpose:
        gsm8k_naive  : compare the last *line* verbatim to the gold number.
        gsm8k_robust : pull the last number, strip $/,/% and trailing punct,
                       compare numerically (42 == 42.0 == "42").
    The gap between them is the demo's "metric != construct" evidence: the model
    can be *right* and still fail a brittle metric.
"""

from __future__ import annotations

import re
import unicodedata

LETTERS = ["A", "B", "C", "D"]


# --------------------------------------------------------------------------- #
# MMLU
# --------------------------------------------------------------------------- #
def extract_letter(text: str) -> str | None:
    """Best-effort extraction of the chosen option letter from model output."""
    if not text:
        return None
    t = unicodedata.normalize("NFKC", text).strip()

    # 1) explicit "answer is (B)" / "answer: B" near the end
    m = re.findall(r"answer\s*(?:is|:)?\s*\(?([A-D])\)?", t, flags=re.IGNORECASE)
    if m:
        return m[-1].upper()
    # 2) a parenthesised or standalone letter on its own
    m = re.findall(r"\(([A-D])\)", t)
    if m:
        return m[-1].upper()
    # 3) first standalone capital letter token
    m = re.findall(r"\b([A-D])\b", t)
    if m:
        return m[0].upper()
    return None


def mmlu_score(item: dict, prediction: str) -> bool:
    return extract_letter(prediction) == item["answer"]


# --------------------------------------------------------------------------- #
# GSM8K
# --------------------------------------------------------------------------- #
_NUM = re.compile(r"-?\$?\d[\d,]*\.?\d*%?")


def _to_float(token: str) -> float | None:
    token = token.strip().strip("$%").replace(",", "")
    try:
        return float(token)
    except ValueError:
        return None


def extract_number_robust(text: str) -> float | None:
    """Last number in the text, currency/percent/comma tolerated."""
    if not text:
        return None
    nums = _NUM.findall(text)
    for tok in reversed(nums):
        val = _to_float(tok)
        if val is not None:
            return val
    return None


def extract_number_naive(text: str) -> str | None:
    """Deliberately brittle: the last non-empty line, verbatim.

    This is the kind of one-line grader people actually ship. "The answer is
    $42." -> "The answer is $42." which will not string-equal "42".
    """
    if not text:
        return None
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines[-1] if lines else None


def gsm8k_score_naive(item: dict, prediction: str) -> bool:
    pred = extract_number_naive(prediction)
    return pred is not None and pred == item["answer"]


def gsm8k_score_robust(item: dict, prediction: str) -> bool:
    pred = extract_number_robust(prediction)
    gold = _to_float(item["answer"])
    return pred is not None and gold is not None and abs(pred - gold) < 1e-6


# --------------------------------------------------------------------------- #
# dispatch
# --------------------------------------------------------------------------- #
def score(item: dict, prediction: str, metric: str = "robust") -> bool:
    """Unified entry. metric in {robust, naive} only affects GSM8K."""
    if item["benchmark"] == "mmlu":
        return mmlu_score(item, prediction)
    if item["benchmark"] == "gsm8k":
        return (gsm8k_score_naive(item, prediction) if metric == "naive"
                else gsm8k_score_robust(item, prediction))
    raise ValueError(f"unknown benchmark {item['benchmark']!r}")
