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

import config  # noqa: E402

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
    tab["params_b"] = tab["system"].map(config.PARAMS_B)
    tab["is_baseline"] = tab["system"].map(_is_baseline)
    return tab.sort_values(["is_baseline", "acc_mmlu"], ascending=[True, False])


def coverage(df: pd.DataFrame) -> pd.DataFrame:
    """Fraction of items where an answer could be extracted (1 - miss rate).

    If one model loses more to extraction failures than another, the comparison
    is not purely about capability -- a fairness / measurement-noise concern that
    is itself a 'what is actually measured' point of the tutorial.
    """
    llm = df[~df["system"].map(_is_baseline)]
    rows = []
    for (s, b), g in llm.groupby(["system", "benchmark"]):
        ex = g["extracted"].astype(str).str.strip()
        miss = (ex == "") | (ex.str.lower() == "nan")
        rows.append({"system": s, "benchmark": b,
                     "coverage": round(1 - miss.mean(), 3), "n": len(g)})
    return pd.DataFrame(rows).sort_values(["benchmark", "system"])


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
           label="naive: first number", color="#bbbbbb")
    ax.bar([i + w / 2 for i in x], tab["acc_robust"], w,
           label="robust: last number", color="#3b7dd8")
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
# 1b  per-subject MMLU (construct variance)
# --------------------------------------------------------------------------- #
def mmlu_by_subject(df: pd.DataFrame) -> pd.DataFrame:
    """Accuracy per MMLU subject per LLM. The spread across subjects is the
    construct-validity point: one 'MMLU accuracy' hides wildly different
    competence (e.g. marketing vs moral_scenarios)."""
    m = df[(df["benchmark"] == "mmlu") & (df["perturbation"] == "")
           & (~df["system"].map(_is_baseline))]
    if m.empty:
        return pd.DataFrame()
    tab = (m.groupby(["subject", "system"])["correct"].mean()
           .unstack(fill_value=0.0))
    tab.insert(0, "n_items", m.groupby("subject")["id"].nunique())
    sort_col = tab.columns[-1]
    return tab.reset_index().sort_values(sort_col, ascending=False)


def fig_mmlu_subjects(tab: pd.DataFrame) -> None:
    if tab.empty:
        return
    sys_cols = [c for c in tab.columns if c not in ("subject", "n_items")]
    best = max(sys_cols, key=lambda c: tab[c].mean())
    t = tab.sort_values(best, ascending=True)
    fig, ax = plt.subplots(figsize=(8, max(6, 0.28 * len(t))))
    ax.barh(t["subject"], t[best], color="#3b7dd8")
    ax.axvline(0.25, ls="--", lw=1, color="#888", label="random (0.25)")
    ax.set_xlim(0, 1.05)
    ax.set_xlabel(f"{best} accuracy")
    ax.set_title("One 'MMLU score' hides large per-subject variance")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(FIG / "mmlu_by_subject.png", dpi=150)
    plt.close()


# --------------------------------------------------------------------------- #
# 2c  ranking (in)stability across benchmarks
# --------------------------------------------------------------------------- #
def _spearman(r1: list[int], r2: list[int]) -> float:
    n = len(r1)
    if n < 2:
        return float("nan")
    d2 = sum((a - b) ** 2 for a, b in zip(r1, r2))
    return 1 - 6 * d2 / (n * (n * n - 1))


def ranking(df: pd.DataFrame) -> pd.DataFrame:
    """Rank the LLMs on MMLU and on GSM8K separately. If the orders disagree,
    no single benchmark (or single number) decides which model is 'best' -- the
    choice of benchmark does."""
    acc = accuracy(df)
    llm = acc[~acc["is_baseline"]].copy()
    if len(llm) < 2:
        return pd.DataFrame()
    llm["rank_mmlu"] = llm["acc_mmlu"].rank(ascending=False, method="min").astype(int)
    llm["rank_gsm8k"] = llm["acc_gsm8k"].rank(ascending=False, method="min").astype(int)
    llm["rank_shift"] = (llm["rank_mmlu"] - llm["rank_gsm8k"]).abs()
    return llm[["system", "params_b", "acc_mmlu", "rank_mmlu", "acc_gsm8k",
                "rank_gsm8k", "rank_shift"]].sort_values("rank_mmlu")


def fig_agreement(tab: pd.DataFrame) -> None:
    if tab.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(tab["acc_mmlu"], tab["acc_gsm8k"], color="#3b7dd8", zorder=3)
    for _, r in tab.iterrows():
        ax.annotate(r["system"], (r["acc_mmlu"], r["acc_gsm8k"]),
                    xytext=(5, 4), textcoords="offset points", fontsize=8)
    lo = min(tab["acc_mmlu"].min(), tab["acc_gsm8k"].min()) - 0.05
    hi = max(tab["acc_mmlu"].max(), tab["acc_gsm8k"].max()) + 0.05
    ax.plot([lo, hi], [lo, hi], ls="--", lw=1, color="#bbb",
            label="equal MMLU/GSM8K")
    ax.set_xlabel("MMLU accuracy")
    ax.set_ylabel("GSM8K accuracy")
    ax.set_title("Same models, two benchmarks: ranking need not agree")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(FIG / "model_agreement.png", dpi=150)
    plt.close()


# --------------------------------------------------------------------------- #
# 3b + errors  qualitative evidence
# --------------------------------------------------------------------------- #
def metric_gap_examples(df: pd.DataFrame) -> pd.DataFrame:
    """GSM8K items the model got right (robust=1) but the naive grader scored
    wrong (naive=0): concrete 'the metric, not the model, failed' cases."""
    g = df[(df["benchmark"] == "gsm8k") & (~df["system"].map(_is_baseline))]
    ex = g[(g["correct"] == 1) & (g["correct_naive"] == 0)]
    return ex[["system", "id", "gold", "extracted", "prediction"]]


def errors(df: pd.DataFrame) -> pd.DataFrame:
    """All wrong answers (robust metric), for qualitative error analysis."""
    e = df[(df["correct"] == 0) & (~df["system"].map(_is_baseline))]
    return e[["system", "benchmark", "subject", "perturbation", "id",
              "gold", "extracted", "prediction"]]


# --------------------------------------------------------------------------- #
def analyze(df: pd.DataFrame) -> None:
    """Run every table + figure on a per-item DataFrame. The frame may be a
    single run or several runs concatenated (incremental model additions)."""
    FIG.mkdir(parents=True, exist_ok=True)
    df = df.fillna("")

    acc = accuracy(df)
    acc.to_csv(RES / "accuracy.csv", index=False)
    fig_saturation(acc)

    cov = coverage(df)
    cov.to_csv(RES / "coverage.csv", index=False)

    gap = metric_gap(df)
    gap.to_csv(RES / "metric_gap.csv", index=False)
    fig_metric_gap(gap)

    rob = robustness(df)
    rob.to_csv(RES / "robustness.csv", index=False)
    fig_robustness(rob)

    subj = mmlu_by_subject(df)
    subj.to_csv(RES / "mmlu_by_subject.csv", index=False)
    fig_mmlu_subjects(subj)

    rank = ranking(df)
    rank.to_csv(RES / "ranking.csv", index=False)
    fig_agreement(rank)

    examples = metric_gap_examples(df)
    examples.to_csv(RES / "metric_gap_examples.csv", index=False)

    errs = errors(df)
    errs.to_csv(RES / "error_analysis.csv", index=False)

    print("=== Accuracy (MMLU vs GSM8K) ===")
    print(acc.to_string(index=False))
    print("\n=== GSM8K metric gap (first-number vs last-number) ===")
    print(gap.to_string(index=False))
    print("\n=== MMLU robustness (original vs perturbed) ===")
    print(rob.to_string(index=False))
    print("\n=== Answer-extraction coverage (1.0 = every item parsed) ===")
    print(cov.to_string(index=False))
    if not rank.empty:
        rho = _spearman(list(rank["rank_mmlu"]), list(rank["rank_gsm8k"]))
        # weight-matched view: drop the small (<3B) rung so a size gap can't be
        # mistaken for a family effect in the ranking comparison
        peer = rank[rank["params_b"] >= 3.0].copy()
        peer["rank_mmlu"] = peer["acc_mmlu"].rank(ascending=False, method="min").astype(int)
        peer["rank_gsm8k"] = peer["acc_gsm8k"].rank(ascending=False, method="min").astype(int)
        rho_peer = _spearman(list(peer["rank_mmlu"]), list(peer["rank_gsm8k"]))
        print(f"\n=== Ranking across benchmarks (Spearman rho all={rho:.2f}, "
              f"~4B peers only={rho_peer:.2f}; 1.0 = identical order) ===")
        print(rank.to_string(index=False))
    if not subj.empty:
        print("\n=== MMLU per-subject spread (top/bottom 5) ===")
        print(pd.concat([subj.head(5), subj.tail(5)]).to_string(index=False))
    print(f"\nmetric-gap examples (right answer, naive grader wrong): "
          f"{len(examples)} rows -> metric_gap_examples.csv")
    print(f"error rows -> error_analysis.csv ({len(errs)})")
    print(f"Figures -> {FIG}")


def main() -> None:
    analyze(pd.read_csv(RES / "per_item.csv"))


if __name__ == "__main__":
    main()
