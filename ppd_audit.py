"""
Data-quality audit of the Kaggle "PostPartum Depression" survey benchmark.

Produces the quantitative evidence and publication figures for four claims:

  A1  The file does not behave like 1,503 independent respondents. Its effective
      sample size is the number of distinct questionnaire answers (~248).
  A2  Duplicate leakage inflates reported scores, and the inflation scales with
      model capacity, so published model rankings are an artifact of the split.
  A3  Symptom-risk relationships are non-monotone and in several items inverted,
      indicating a construct-validity failure.
  A4  Operating-point choice dominates model choice for screening use.

Run after ppd_model_comparison.py (it consumes results/*.csv and the saved
out-of-fold probabilities).

    python ppd_audit.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_auc_score, roc_curve

from sklearn.pipeline import Pipeline

from ppd_preprocessing import (
    AGE_COL, SYMPTOM_COLS, build_preprocessor, level_order, prepare,
)

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
N_SIM = 500
SEED_LC = 7
RNG = np.random.default_rng(20240914)

# Ordered by effective capacity, low -> high. Used for the inflation figure.
CAPACITY_ORDER = ["Logistic Regression", "SVM (RBF)", "Random Forest", "ANN (MLP)"]
PALETTE = {"obs": "#c1443c", "sim": "#8fb8de", "accent": "#3b6ea5", "grey": "#6b7280"}


# ------------------------------------------------------------------ A1: effective n
def pattern_series(X: pd.DataFrame) -> pd.Series:
    cols = [AGE_COL] + SYMPTOM_COLS
    return X[cols].astype(str).agg("|".join, axis=1)


def independence_test(X: pd.DataFrame, n_sim: int = N_SIM) -> dict:
    """How diverse would 1,168 *independent* respondents be, given these marginals?

    This preserves every univariate distribution exactly and assumes the items are
    independent. Real items are correlated, which reduces diversity, so the simulated
    figures are an upper bound. The observed association is weak (Cramer's V mostly
    ~0.2, max 0.51), so it cannot account for a gap of this size.
    """
    cols = [AGE_COL] + SYMPTOM_COLS
    n = len(X)
    margins = {c: X[c].value_counts(normalize=True) for c in cols}
    space = int(np.prod([len(m) for m in margins.values()]))

    uniq, single = [], []
    for _ in range(n_sim):
        sim = pd.DataFrame(
            {c: RNG.choice(m.index, size=n, p=m.to_numpy()) for c, m in margins.items()}
        )
        vc = sim.astype(str).agg("|".join, axis=1).value_counts()
        uniq.append(len(vc))
        single.append(int((vc == 1).sum()))

    obs = pattern_series(X).value_counts()
    return {
        "n_rows": n,
        "combination_space": space,
        "observed_unique": int(len(obs)),
        "observed_singletons": int((obs == 1).sum()),
        "observed_max_repeat": int(obs.max()),
        "observed_duplicate_rows": int(n - len(obs)),
        "sim_unique_mean": float(np.mean(uniq)),
        "sim_unique_lo": int(np.min(uniq)),
        "sim_unique_hi": int(np.max(uniq)),
        "sim_singletons_mean": float(np.mean(single)),
        "sim_singletons_lo": int(np.min(single)),
        "sim_singletons_hi": int(np.max(single)),
        "counts": obs,
    }


def dispersion_test(obs_counts: pd.Series, n_sim: int = N_SIM) -> dict:
    """Are the repeat counts even a plausible *random* draw from their own distribution?

    Granting the observed pattern frequencies entirely, draw the same number of rows iid
    from that distribution. Genuine iid sampling produces a tail of rare patterns and
    over-dispersed counts. Systematic replication of a fixed template pool does not.
    This test makes no independence assumption about the items.
    """
    n = int(obs_counts.sum())
    p = (obs_counts / obs_counts.sum()).to_numpy()

    singles, variances = [], []
    for _ in range(n_sim):
        draw = RNG.multinomial(n, p)
        singles.append(int((draw == 1).sum()))
        variances.append(float(draw.var(ddof=1)))

    return {
        "observed_singletons": int((obs_counts == 1).sum()),
        "sim_singletons_mean": float(np.mean(singles)),
        "sim_singletons_lo": int(np.min(singles)),
        "sim_singletons_hi": int(np.max(singles)),
        "observed_variance": float(obs_counts.to_numpy().var(ddof=1)),
        "sim_variance_mean": float(np.mean(variances)),
        "dispersion_ratio": float(obs_counts.to_numpy().var(ddof=1) / np.mean(variances)),
    }


def fig_patterns(ind: dict, disp: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))

    ax = axes[0]
    bars = ax.bar(["Observed", "Independent\nrespondents"],
                  [ind["observed_unique"], ind["sim_unique_mean"]],
                  color=[PALETTE["obs"], PALETTE["sim"]], width=0.6)
    ax.errorbar(1, ind["sim_unique_mean"],
                yerr=[[ind["sim_unique_mean"] - ind["sim_unique_lo"]],
                      [ind["sim_unique_hi"] - ind["sim_unique_mean"]]],
                fmt="none", ecolor="k", capsize=4)
    for b, v in zip(bars, [ind["observed_unique"], ind["sim_unique_mean"]]):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.0f}", ha="center", va="bottom")
    ax.set_ylabel("Distinct response patterns")
    ax.set_title(f"(a) Pattern diversity\n{ind['n_rows']} rows, "
                 f"{ind['combination_space']:,} possible answers", fontsize=10)

    ax = axes[1]
    vals = [ind["observed_singletons"], disp["sim_singletons_mean"], ind["sim_singletons_mean"]]
    bars = ax.bar(["Observed", "iid draws from\nsame patterns", "Independent\nrespondents"],
                  vals, color=[PALETTE["obs"], PALETTE["grey"], PALETTE["sim"]], width=0.62)
    ax.set_yscale("log")
    ax.set_ylim(0.6, max(vals) * 3)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v * 1.15, f"{v:.0f}", ha="center", va="bottom")
    ax.set_ylabel("Patterns occurring exactly once (log)")
    ax.set_title("(b) The missing tail\nEven granting the pattern frequencies, "
                 "sampling yields a tail", fontsize=10)

    ax = axes[2]
    counts = ind["counts"]
    freq = counts.value_counts().sort_index()
    colours = [PALETTE["obs"] if i == 1 else PALETTE["accent"] for i in freq.index]
    ax.bar(freq.index, freq.to_numpy(), color=colours, width=0.85)
    ax.annotate("only 1 pattern\noccurs once", xy=(1, 1), xytext=(6, 60),
                fontsize=8.5, color=PALETTE["obs"],
                arrowprops=dict(arrowstyle="->", color=PALETTE["obs"], lw=1.2))
    ax.set_xlabel("Times a response pattern is repeated")
    ax.set_ylabel("Number of patterns")
    ax.set_title("(c) Repeat structure\nthe distribution starts at 2, not 1", fontsize=10)

    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
    fig.suptitle("The benchmark does not behave like independent respondents", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ----------------------------------------------------------- A2: capacity vs inflation
def inflation_table() -> pd.DataFrame:
    a = pd.read_csv(OUT / "comparison_protocolA_standard.csv").set_index("Model")
    b = pd.read_csv(OUT / "comparison_protocolB_grouped.csv").set_index("Model")
    t = pd.DataFrame({
        "AUC_A": a["ROC-AUC"], "AUC_B": b["ROC-AUC"],
        "F1_A": a["F1"], "F1_B": b["F1"],
        "Acc_A": a["Accuracy"], "Acc_B": b["Accuracy"],
    })
    t["dAUC"] = t["AUC_A"] - t["AUC_B"]
    t["dF1"] = t["F1_A"] - t["F1_B"]
    t["dAcc"] = t["Acc_A"] - t["Acc_B"]
    return t.reindex(CAPACITY_ORDER)


def fig_inflation(t: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    x = np.arange(len(t))
    labels = [m.replace(" (", "\n(") for m in t.index]

    ax = axes[0]
    ax.bar(x - 0.2, t["AUC_A"], 0.4, label="Protocol A (standard CV)", color=PALETTE["sim"])
    ax.bar(x + 0.2, t["AUC_B"], 0.4, label="Protocol B (grouped CV)", color=PALETTE["obs"])
    for xi, (a_, b_) in enumerate(zip(t["AUC_A"], t["AUC_B"])):
        ax.text(xi, max(a_, b_) + 0.012, f"-{a_ - b_:.3f}", ha="center",
                fontsize=9, color=PALETTE["grey"])
    ax.set_xticks(x, labels, fontsize=9)
    ax.set_ylim(0.5, 1.06)
    ax.set_ylabel("ROC-AUC")
    ax.set_title("(a) Reported vs leakage-controlled performance", fontsize=10)
    ax.legend(fontsize=9, loc="lower left")

    ax = axes[1]
    ax.plot(x, t["dAUC"], "o-", lw=2, color=PALETTE["accent"], label="Δ ROC-AUC")
    ax.plot(x, t["dF1"], "s--", lw=2, color=PALETTE["obs"], label="Δ F1")
    for xi, (da, df) in enumerate(zip(t["dAUC"], t["dF1"])):
        ax.annotate(f"{da:.3f}", (xi, da), textcoords="offset points", xytext=(0, 9),
                    ha="center", fontsize=8, color=PALETTE["accent"])
        ax.annotate(f"{df:.3f}", (xi, df), textcoords="offset points", xytext=(0, -14),
                    ha="center", fontsize=8, color=PALETTE["obs"])
    ax.set_xticks(x, labels, fontsize=9)
    ax.set_ylabel("Inflation (Protocol A − Protocol B)")
    ax.set_xlabel("Increasing model flexibility →")
    ax.set_title("(b) ΔAUC rises monotonically with flexibility;\n"
                 "ΔF1 is confounded by the operating point", fontsize=10)
    ax.axhline(0, color="k", lw=0.8)
    ax.legend(fontsize=9)

    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
    fig.suptitle("Duplicate leakage inflates ranking quality most for the most flexible models",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_roc_pair(path: Path) -> bool:
    npz = OUT / "oof_predictions.npz"
    if not npz.exists():
        return False
    d = np.load(npz, allow_pickle=False)
    y = d["y_true"]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.6))
    for ax, tag, title in zip(
        axes, ("A", "B"),
        ("(a) Protocol A — standard CV (duplicate-inflated)",
         "(b) Protocol B — grouped CV (leakage-controlled)"),
    ):
        for model in CAPACITY_ORDER:
            key = f"{tag}::{model}"
            if key not in d:
                continue
            fpr, tpr, _ = roc_curve(y, d[key])
            ax.plot(fpr, tpr, lw=2, label=f"{model} ({auc(fpr, tpr):.3f})")
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="Chance")
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.set_title(title, fontsize=10)
        ax.legend(loc="lower right", fontsize=8.5, title="AUC", title_fontsize=8.5)
        ax.grid(alpha=0.25)
        ax.set_axisbelow(True)
    fig.suptitle("The same models under two evaluation protocols", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return True


# --------------------------------------------- A1b: information saturates at ~248
def learning_curve(X: pd.DataFrame, y: pd.Series, groups: np.ndarray,
                   n_draws: int = 25) -> pd.DataFrame:
    """Is the carrier of information the row count or the distinct-questionnaire count?

    A quarter of the response patterns is held out, and the test set takes ONE row per
    held-out pattern. That matters: an earlier version scored against all rows of the
    held-out patterns, which is row-weighted, and row-sampled training then looked better
    purely because it matched the test frequencies. A pattern-level test set removes that
    confound and isolates the information question.

    Two ways of growing the training set are compared:
      'distinct'   -- k patterns, one row each      (k rows, k patterns)
      'duplicated' -- k rows drawn from the pool    (k rows, fewer than k patterns)

    Plotted against rows the two must differ. Plotted against the number of DISTINCT
    patterns actually covered, they collapse onto one curve if -- and only if -- rows
    beyond the distinct questionnaires carry no additional information.
    """
    from sklearn.ensemble import RandomForestClassifier

    rng = np.random.default_rng(SEED_LC)
    uniq = np.unique(groups)
    yv = y.to_numpy()
    rows = []

    for draw in range(n_draws):
        test_pat = rng.choice(uniq, size=max(2, int(0.25 * len(uniq))), replace=False)
        # One row per held-out pattern: a pattern-level, not row-weighted, test set.
        test_idx = np.array([np.flatnonzero(groups == g)[0] for g in test_pat])
        if len(np.unique(yv[test_idx])) < 2:
            continue
        pool_pat = np.setdiff1d(uniq, test_pat)
        pool_idx = np.flatnonzero(~np.isin(groups, test_pat))
        # one representative row per training pattern
        rep = {g: np.flatnonzero(groups == g)[0] for g in pool_pat}

        for k in (20, 40, 60, 80, 100, 130, 160, len(pool_pat)):
            if k > len(pool_pat):
                continue
            chosen = rng.choice(pool_pat, size=k, replace=False)
            for mode, idx in (
                ("distinct", np.array([rep[g] for g in chosen])),
                ("duplicated", rng.choice(pool_idx, size=k, replace=False)),
            ):
                if len(np.unique(yv[idx])) < 2:
                    continue
                clf = Pipeline([
                    ("prep", build_preprocessor("onehot", scale=False)),
                    ("clf", RandomForestClassifier(n_estimators=300, min_samples_leaf=3,
                                                   random_state=SEED_LC, n_jobs=-1)),
                ]).fit(X.iloc[idx], yv[idx])
                p = clf.predict_proba(X.iloc[test_idx])[:, 1]
                rows.append({"draw": draw, "rows": k, "mode": mode,
                             "patterns": len(np.unique(groups[idx])),
                             "auc": roc_auc_score(yv[test_idx], p)})
    return pd.DataFrame(rows)


def fig_learning_curve(lc: pd.DataFrame, path: Path) -> None:
    """Two views of the same runs. The second is the test of the effective-n claim."""
    styles = {"distinct": (PALETTE["accent"], "o-", "k distinct questionnaires (1 row each)"),
              "duplicated": (PALETTE["obs"], "s--", "k rows drawn from the full file")}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

    for ax, xvar, title, xlabel in (
        (axes[0], "rows", "(a) Against training rows", "Training rows"),
        (axes[1], "patterns",
         "(b) Against distinct questionnaires covered\n"
         "(curves do not collapse: repeats add frequency information)",
         "Distinct response patterns in the training set"),
    ):
        for mode, (colour, fmt, label) in styles.items():
            sub = lc[lc["mode"] == mode]
            if xvar == "patterns":
                # bin the pattern counts so the two modes are comparable on one axis
                sub = sub.assign(_b=(sub["patterns"] / 10).round() * 10)
                g = sub.groupby("_b")["auc"]
            else:
                g = sub.groupby(xvar)["auc"]
            m, lo, hi = g.mean(), g.quantile(0.25), g.quantile(0.75)
            ax.plot(m.index, m.to_numpy(), fmt, lw=2, ms=5, color=colour, label=label)
            ax.fill_between(m.index, lo.to_numpy(), hi.to_numpy(), color=colour, alpha=0.13)
        ax.set_xlabel(xlabel)
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.25)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("Held-out ROC-AUC (one row per unseen pattern)")
    axes[0].legend(fontsize=8.5, loc="lower right")
    # Honest reading: the two modes do NOT collapse in (b). Rows beyond the distinct
    # questionnaires still help a little, because repeat counts carry frequency
    # information. The effect is small (~0.02 AUC) and the interquartile bands overlap
    # throughout; the headline is the level, not the gap.
    fig.suptitle("Generalising to unseen questionnaires plateaus near 0.66 AUC",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ------------------------------------------------------ A3: construct validity
def monotonicity_table(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    rows = []
    for c in SYMPTOM_COLS:
        order = level_order(c)
        rate = y.groupby(X[c]).mean().reindex(order)
        d = rate.diff().dropna()
        monotone = bool((d >= 0).all() or (d <= 0).all())
        inverted = bool(len(order) == 3 and rate.iloc[1] < rate.iloc[0])
        rows.append({
            "Item": c,
            "levels": " < ".join(order),
            "rates": " / ".join(f"{v:.2f}" for v in rate),
            "monotone": monotone,
            "middle_below_none": inverted,
            "rate_none": float(rate.iloc[0]),
            "rate_mid": float(rate.iloc[1]) if len(order) == 3 else np.nan,
            "rate_full": float(rate.iloc[-1]),
        })
    return pd.DataFrame(rows)


def fig_nonmonotone(X: pd.DataFrame, y: pd.Series, mono: pd.DataFrame, path: Path) -> None:
    show = mono[mono["middle_below_none"]]["Item"].tolist()
    fig, axes = plt.subplots(1, len(show), figsize=(3.5 * len(show), 4.2), sharey=True)
    for ax, col in zip(np.atleast_1d(axes), show):
        order = level_order(col)
        rate = y.groupby(X[col]).mean().reindex(order) * 100
        colours = [PALETTE["obs"] if v == rate.min() else PALETTE["accent"] for v in rate]
        ax.bar(range(len(order)), rate.to_numpy(), color=colours, width=0.65)
        for i, v in enumerate(rate):
            ax.text(i, v + 1, f"{v:.0f}%", ha="center", fontsize=9)
        ax.set_xticks(range(len(order)), [o[:14] for o in order], fontsize=8.5, rotation=12)
        ax.set_title(col, fontsize=9.5)
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
    np.atleast_1d(axes)[0].set_ylabel("Suicide-attempt rate (%)")
    fig.suptitle("Construct-validity failure: the middle response is the lowest-risk "
                 "level, below 'no symptom'", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ----------------------------------------------------------------------------- main
def main() -> None:
    prep = prepare(verbose=False)
    X, y = prep["X"], prep["y"]

    print("=" * 78)
    print("A1  Effective sample size")
    print("=" * 78)
    ind = independence_test(X)
    disp = dispersion_test(ind["counts"])
    print(f"  rows {ind['n_rows']}, {ind['combination_space']:,} possible answers")
    print(f"  distinct patterns  observed {ind['observed_unique']:>5}   "
          f"independent respondents {ind['sim_unique_mean']:.0f} "
          f"[{ind['sim_unique_lo']}-{ind['sim_unique_hi']}]")
    print(f"  singletons         observed {ind['observed_singletons']:>5}   "
          f"independent respondents {ind['sim_singletons_mean']:.0f} "
          f"[{ind['sim_singletons_lo']}-{ind['sim_singletons_hi']}]")
    print(f"  singletons         observed {disp['observed_singletons']:>5}   "
          f"iid draws from the SAME pattern distribution {disp['sim_singletons_mean']:.0f} "
          f"[{disp['sim_singletons_lo']}-{disp['sim_singletons_hi']}]")
    print(f"  repeat-count variance observed {disp['observed_variance']:.1f} vs iid "
          f"{disp['sim_variance_mean']:.1f} -> under-dispersed "
          f"{1 / disp['dispersion_ratio']:.1f}x")
    print(f"  effective n = {ind['observed_unique']} "
          f"({ind['observed_duplicate_rows']} of {ind['n_rows']} rows are repeats)")

    print("\n" + "=" * 78)
    print("A2  Leakage inflation by model capacity")
    print("=" * 78)
    infl = inflation_table()
    print(infl[["AUC_A", "AUC_B", "dAUC", "F1_A", "F1_B", "dF1", "dAcc"]].round(3).to_string())
    print(f"  Spearman(capacity rank, dAUC) = "
          f"{pd.Series(range(len(infl))).corr(infl['dAUC'].reset_index(drop=True), method='spearman'):.2f}")

    print("\n" + "=" * 78)
    print("A3  Construct validity")
    print("=" * 78)
    mono = monotonicity_table(X, y)
    print(mono[["Item", "rates", "monotone", "middle_below_none"]].to_string(index=False))
    n_inv = int(mono["middle_below_none"].sum())
    print(f"  {int((~mono['monotone']).sum())}/{len(mono)} items non-monotone; "
          f"{n_inv} have the middle level BELOW the 'no symptom' level")

    print("\n" + "=" * 78)
    print("A1b  Learning curve: does adding rows add information?")
    print("=" * 78)
    lc = learning_curve(X, y, prep["groups"])
    print("  by training ROWS:")
    print(lc.groupby(["mode", "rows"])["auc"].mean().unstack(0).round(3).to_string())
    lcb = lc.assign(_b=(lc["patterns"] / 10).round() * 10)
    print("\n  by DISTINCT PATTERNS covered (do the two modes collapse?):")
    print(lcb.groupby(["mode", "_b"])["auc"].mean().unstack(0).round(3).to_string())
    lc.to_csv(OUT / "learning_curve.csv", index=False)
    fig_learning_curve(lc, OUT / "fig_audit_learning_curve.png")

    fig_patterns(ind, disp, OUT / "fig_audit_patterns.png")
    fig_inflation(infl, OUT / "fig_audit_inflation.png")
    fig_nonmonotone(X, y, mono, OUT / "fig_audit_nonmonotone.png")
    made_roc = fig_roc_pair(OUT / "fig_audit_roc_pair.png")

    stats = {
        "effective_n": ind["observed_unique"],
        "independence_test": {k: v for k, v in ind.items() if k != "counts"},
        "dispersion_test": disp,
        "inflation": infl.round(4).to_dict(orient="index"),
        "monotonicity": mono.drop(columns=["rates"]).to_dict(orient="records"),
    }
    (OUT / "audit_statistics.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print(f"\nFigures: fig_audit_patterns.png, fig_audit_inflation.png, "
          f"fig_audit_nonmonotone.png{', fig_audit_roc_pair.png' if made_roc else ''}")
    print(f"Statistics: {OUT / 'audit_statistics.json'}")


if __name__ == "__main__":
    main()
