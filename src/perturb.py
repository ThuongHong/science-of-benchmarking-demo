"""Input perturbations for MMLU items, to probe robustness (SuperGLUE point).

A benchmark number is only trustworthy if it survives meaning-preserving (or
label-preserving) noise. We generate twin items so accuracy on the original vs
the perturbed copy is an apples-to-apples paired comparison.

Perturbations
-------------
option_shuffle : permute the four options and remap the gold letter. Same
                 semantics, different surface position -> exposes positional /
                 "always-answer-A" bias.
distractor     : append a plausible-looking 5th option "E. None of the above".
                 The correct letter is unchanged, but the model now has to
                 resist a tempting catch-all -> exposes shallow option matching.
"""

from __future__ import annotations

import random

LETTERS = ["A", "B", "C", "D", "E"]


def option_shuffle(item: dict, seed: int = 0) -> dict:
    rng = random.Random(f"{item['id']}-{seed}")
    order = list(range(len(item["choices"])))
    rng.shuffle(order)
    new_choices = [item["choices"][i] for i in order]
    new_index = order.index(item["answer_index"])
    return {
        **item,
        "id": f"{item['id']}__shuffle",
        "orig_id": item["id"],
        "perturbation": "option_shuffle",
        "choices": new_choices,
        "answer_index": new_index,
        "answer": LETTERS[new_index],
    }


def distractor(item: dict) -> dict:
    new_choices = list(item["choices"]) + ["None of the above"]
    return {
        **item,
        "id": f"{item['id']}__distractor",
        "orig_id": item["id"],
        "perturbation": "distractor",
        "choices": new_choices,
        # gold letter/index unchanged: the real answer is still present
    }


def build_perturbations(mmlu_items: list[dict], n: int = 50,
                        seed: int = 0) -> list[dict]:
    """One perturbation per source item, alternating the two kinds."""
    out = []
    for k, item in enumerate(mmlu_items[:n]):
        out.append(option_shuffle(item, seed) if k % 2 == 0
                   else distractor(item))
    return out
