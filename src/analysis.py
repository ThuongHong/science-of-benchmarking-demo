"""Turn results/per_item.csv into the tutorial's five evidence tables + figures.

  1. saturation   accuracy.csv / saturation.png
       Per system, accuracy on MMLU vs GSM8K. A benchmark "saturates" when the
       strong systems bunch near the top and stop separating (GLUE -> SuperGLUE).
  2. baselines    folded into accuracy.csv + overall.png
       Real models must clear random / majority by a wide margin, else the score
       is not measuring competence (Measuring what Matters).
  3. metric gap   metric_gap.csv / metric_gap.png
       GSM8K naive vs robust grader. n_robust_only = right answers a brittle
       grader throws away (metric != construct).
  4. robustness   robustness.csv / robustness.png
       MMLU original vs perturbed twin, paired -> accuracy drop (SuperGLUE).
  5. contamination is discussed qualitatively in the README, not computed here.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
FIG = RES / "figures"


def _is_baseline(s: str) -> bool:
    return s.startswith("baseline")


# --------------------------------------------------------------------------- #
# 1 + 2  accuracy / saturation / baselines
# --------------------------------------------------------------------------- #
def accuracy(df: pd.DataFrame) -> pd.DataFrame:
    main = df[df["perturbation"] == ""]
    tab = (main.groupby(["system", "benchmark"])["correct"].mean()
           .unstack(fill_value=0.0).reset_index())
    tab = tab.rename(columns={"mmlu": "acc_mmlu", "gsm8k": "acc_gsm8k"})
    tab["is_baseline"] = tab["system"].map(_is_baseline)
    return tab.sort_values(["is_baseline", "acc_mmlu"], ascending=[True, False])


def fig_saturation(tab: pd.DataFrame) -> None:
    x = range(len(tab))
    w = 0.38
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar([i - w / 2 for i in x], tab["acc_mmlu"], w, label="MMLU",
           color="#9ecae1")
    ax.bar([i + w / 2 for i in x], tab["acc_gsm8k"], w, label="GSM8K",
           color="#d8533b")
    ax.axhline(0.25, ls="--", lw=1, color="#888",
               label="MMLU random (0.25)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(tab["system"], rotation=20, ha="right")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("Accuracy on real benchmarks: MMLU (knowledge) vs GSM8K (reasoning)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIG / "saturation.png", dpi=150)
    plt.close()


# --------------------------------------------------------------------------- #
# 3  metric brittleness (GSM8K)
# --------------------------------------------------------------------------- #
def metric_gap(df: pd.DataFrame) -> pd.DataFrame:
    g = df[(df["benchmark"] == "gsm8k") & (~df["system"].map(_is_baseline))]
    rows = []
    for sys, sub in g.groupby("system"):
        robust_only = ((sub["correct"] == 1) & (sub["correct_naive"] == 0)).sum()
        rows.append({
            "system": sys,
            "acc_naive": sub["correct_naive"].mean(),
            "acc_robust": sub["correct"].mean(),
            "n_robust_only": int(robust_only),
            "n": len(sub),
        })
    return pd.DataFrame(rows).sort_values("acc_robust", ascending=False)


def fig_metric_gap(tab: pd.DataFrame) -> None:
    if tab.empty:
        return
    x = range(len(tab))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar([i - w / 2 for i in x], tab["acc_naive"], w,
           label="naive last-line grader", color="#bbbbbb")
    ax.bar([i + w / 2 for i in x], tab["acc_robust"], w,
           label="robust numeric grader", color="#3b7dd8")
    ax.set_xticks(list(x))
    ax.set_xticklabels(tab["system"], rotation=15, ha="right")
    ax.set_ylabel("GSM8K accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("Same answers, two graders: a brittle metric understates skill")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIG / "metric_gap.png", dpi=150)
    plt.close()


# --------------------------------------------------------------------------- #
# 4  robustness (MMLU original vs perturbed)
# --------------------------------------------------------------------------- #
def robustness(df: pd.DataFrame) -> pd.DataFrame:
    pert = df[df["perturbation"] != ""].copy()
    if pert.empty:
        return pd.DataFrame()
    twin_ids = set(pert["orig_id"])
    orig = df[(df["perturbation"] == "") & (df["id"].isin(twin_ids))]

    o = orig.groupby("system")["correct"].mean().rename("acc_original")
    p = pert.groupby("system")["correct"].mean().rename("acc_perturbed")
    tab = pd.concat([o, p], axis=1).reset_index()
    tab["drop"] = tab["acc_original"] - tab["acc_perturbed"]
    tab = tab[~tab["system"].map(_is_baseline)]
    return tab.sort_values("drop", ascending=False)


def fig_robustness(tab: pd.DataFrame) -> None:
    if tab.empty:
        return
    x = range(len(tab))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar([i - w / 2 for i in x], tab["acc_original"], w, label="original",
           color="#74c476")
    ax.bar([i + w / 2 for i in x], tab["acc_perturbed"], w, label="perturbed",
           color="#fd8d3c")
    ax.set_xticks(list(x))
    ax.set_xticklabels(tab["system"], rotation=15, ha="right")
    ax.set_ylabel("MMLU accuracy (paired items)")
    ax.set_ylim(0, 1.05)
    ax.set_title("Robustness: accuracy drop under option shuffle / distractor")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIG / "robustness.png", dpi=150)
    plt.close()


# --------------------------------------------------------------------------- #
def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(RES / "per_item.csv").fillna("")

    acc = accuracy(df)
    acc.to_csv(RES / "accuracy.csv", index=False)
    fig_saturation(acc)

    gap = metric_gap(df)
    gap.to_csv(RES / "metric_gap.csv", index=False)
    fig_metric_gap(gap)

    rob = robustness(df)
    rob.to_csv(RES / "robustness.csv", index=False)
    fig_robustness(rob)

    print("=== Accuracy (MMLU vs GSM8K) ===")
    print(acc.to_string(index=False))
    print("\n=== GSM8K metric gap (naive vs robust) ===")
    print(gap.to_string(index=False))
    print("\n=== MMLU robustness (original vs perturbed) ===")
    print(rob.to_string(index=False))
    print(f"\nFigures -> {FIG}")


if __name__ == "__main__":
    main()
