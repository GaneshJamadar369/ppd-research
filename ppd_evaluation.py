"""
Publication-grade evaluation for the PPD structured branch.

Upgrades the baseline comparison to something defensible under review:

  * Repeated nested CV (R repeats x 5 outer folds, inner 3-fold tuning) instead of a
    single split, because 248 effective samples make one 5-fold estimate very noisy.
  * Cluster bootstrap confidence intervals. Rows are resampled by RESPONSE PATTERN,
    not individually, so the intervals respect the duplication in the data.
  * Paired bootstrap tests between models, so claimed rankings are supported. The
    current headline gap (RF 0.890 vs SVM 0.891 AUC) is almost certainly noise.
  * Trivial baselines: majority class, and a single-feature logistic regression on the
    composite severity score. If the engineered score alone matches a tuned ensemble,
    the ensemble is adding nothing and the paper should say so.
  * Gradient boosting added to the model set.
  * Calibration (Brier, ECE, reliability curves) and decision-curve analysis, because
    the paper recommends net-benefit reporting for screening and must therefore do it.
  * A third protocol: deduplicated data (one row per distinct questionnaire).

Protocols
  A  standard stratified CV                    -- literature-comparable, leaky
  B  grouped CV on the response pattern        -- leakage-controlled, primary
  C  deduplicated (unique questionnaires only) -- the honest small dataset

    python ppd_evaluation.py [--repeats 3] [--encoding onehot]
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, brier_score_loss, confusion_matrix, f1_score,
    matthews_corrcoef, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from ppd_preprocessing import SCORE_COL, build_preprocessor, prepare

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

SEED = 42
N_BOOT = 2000
OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)
ENCODING = "onehot"

PALETTE = {"obs": "#c1443c", "sim": "#8fb8de", "accent": "#3b6ea5", "grey": "#6b7280",
           "green": "#4f8a5b", "purple": "#7c6bad"}
MODEL_COLOURS = {
    "Majority class": "#9aa5b1", "Severity score only": "#4f8a5b",
    "Logistic Regression": "#3b6ea5", "SVM (RBF)": "#c1443c",
    "Random Forest": "#7c6bad", "Gradient Boosting": "#c98a1e", "ANN (MLP)": "#2f8f8a",
}


# ------------------------------------------------------------------------- models
def score_only_pipeline() -> Pipeline:
    """Single-feature baseline: logistic regression on the composite severity score."""
    return Pipeline([
        ("encode", ColumnTransformer(
            [("score", SimpleImputer(strategy="median"), [SCORE_COL])])),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, random_state=SEED)),
    ])


def build_models() -> dict[str, tuple[Pipeline, dict]]:
    pre = lambda scale: build_preprocessor(ENCODING, scale=scale)  # noqa: E731
    return {
        "Majority class": (
            Pipeline([("clf", DummyClassifier(strategy="prior"))]), {}),
        "Severity score only": (score_only_pipeline(), {"clf__C": [0.1, 1.0, 10.0]}),
        "Logistic Regression": (
            Pipeline([("prep", pre(True)),
                      ("clf", LogisticRegression(max_iter=5000, random_state=SEED))]),
            {"clf__C": [0.01, 0.1, 1.0, 10.0], "clf__class_weight": [None, "balanced"]}),
        "SVM (RBF)": (
            Pipeline([("prep", pre(True)),
                      ("clf", SVC(kernel="rbf", probability=True, random_state=SEED))]),
            {"clf__C": [0.1, 1.0, 10.0], "clf__gamma": ["scale", 0.1]}),
        "Random Forest": (
            Pipeline([("prep", pre(False)),
                      ("clf", RandomForestClassifier(n_estimators=400, random_state=SEED,
                                                     n_jobs=-1))]),
            {"clf__max_depth": [None, 6], "clf__min_samples_leaf": [1, 3, 10]}),
        "Gradient Boosting": (
            Pipeline([("prep", pre(False)),
                      ("clf", HistGradientBoostingClassifier(random_state=SEED))]),
            {"clf__max_leaf_nodes": [15, 31], "clf__learning_rate": [0.05, 0.1]}),
        "ANN (MLP)": (
            Pipeline([("prep", pre(True)),
                      ("clf", MLPClassifier(max_iter=1500, early_stopping=True,
                                            n_iter_no_change=25, random_state=SEED))]),
            {"clf__hidden_layer_sizes": [(32,), (64, 32)], "clf__alpha": [1e-4, 1e-2]}),
    }


# --------------------------------------------------------------------- CV machinery
def run_repeated_cv(X, y, groups, protocol: str, repeats: int) -> dict:
    """Repeated nested CV. Returns per-model per-fold metrics and averaged OOF probs."""
    yv = y.to_numpy()
    grouped = protocol == "B"
    models = build_models()
    out: dict[str, dict] = {}

    print(f"\n{'=' * 78}\nPROTOCOL {protocol}  ({len(X)} rows, "
          f"{len(np.unique(groups)) if groups is not None else len(X)} groups, "
          f"{repeats} repeats)\n{'=' * 78}")

    for name, (pipe, grid) in models.items():
        fold_rows, oof_reps = [], []
        for rep in range(repeats):
            rs = SEED + rep * 101
            if grouped:
                outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=rs)
                inner = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=rs)
                splits = outer.split(X, yv, groups)
            else:
                outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=rs)
                inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=rs)
                splits = outer.split(X, yv)

            oof = np.zeros(len(X))
            for tr, te in splits:
                if grid:
                    search = GridSearchCV(pipe, grid, scoring="roc_auc", cv=inner, n_jobs=-1)
                    if grouped:
                        search.fit(X.iloc[tr], yv[tr], groups=groups[tr])
                    else:
                        search.fit(X.iloc[tr], yv[tr])
                    est = search.best_estimator_
                else:
                    est = pipe.fit(X.iloc[tr], yv[tr])
                p = est.predict_proba(X.iloc[te])[:, 1]
                oof[te] = p
                fold_rows.append(point_metrics(yv[te], (p >= 0.5).astype(int), p))
            oof_reps.append(oof)

        folds = pd.DataFrame(fold_rows)
        mean_prob = np.mean(oof_reps, axis=0)
        out[name] = {"folds": folds, "prob": mean_prob}
        m = point_metrics(yv, (mean_prob >= 0.5).astype(int), mean_prob)
        print(f"  {name:<22} AUC {m['ROC-AUC']:.3f}  Acc {m['Accuracy']:.3f}  "
              f"F1 {m['F1']:.3f}  Brier {m['Brier']:.3f}", flush=True)
    return out


def point_metrics(y_true, y_pred, y_prob) -> dict[str, float]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    single = len(np.unique(y_true)) < 2
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "ROC-AUC": 0.5 if single else roc_auc_score(y_true, y_prob),
        "Specificity": tn / (tn + fp) if (tn + fp) else 0.0,
        "MCC": matthews_corrcoef(y_true, y_pred),
        "Brier": brier_score_loss(y_true, y_prob),
    }


# --------------------------------------------------- pattern-weighted (unit = person)
def pattern_aggregate(y, prob, groups) -> tuple[np.ndarray, np.ndarray]:
    """Collapse rows to one point per distinct questionnaire.

    Grouped CV stops a duplicated questionnaire from spanning a split, but the test fold
    is still ROW-weighted: a pattern present 33 times is scored 33 times. If the repeats
    are replication artifacts rather than 33 real respondents, that silently reweights
    the benchmark toward whichever questionnaires happen to be duplicated most. Scoring
    each distinct questionnaire once is the estimate that transfers to a new respondent.
    """
    df = pd.DataFrame({"g": groups, "y": y, "p": prob})
    agg = df.groupby("g").agg(y=("y", lambda s: int(s.mean() >= 0.5)), p=("p", "mean"))
    return agg["y"].to_numpy(), agg["p"].to_numpy()


def pattern_boot_indices(n_patterns: int, n_boot: int, rng) -> list[np.ndarray]:
    return [rng.choice(n_patterns, size=n_patterns, replace=True) for _ in range(n_boot)]


# ------------------------------------------------------------- cluster bootstrap CIs
def cluster_boot_indices(groups: np.ndarray, n_boot: int, rng) -> list[np.ndarray]:
    """Resample whole response-pattern clusters, so CIs respect the duplication."""
    uniq = np.unique(groups)
    members = {g: np.flatnonzero(groups == g) for g in uniq}
    out = []
    for _ in range(n_boot):
        drawn = rng.choice(uniq, size=len(uniq), replace=True)
        out.append(np.concatenate([members[g] for g in drawn]))
    return out


def boot_ci(y, prob, idx_sets, metric="ROC-AUC") -> tuple[float, float, float]:
    vals = []
    for idx in idx_sets:
        yy = y[idx]
        if len(np.unique(yy)) < 2:
            continue
        pp = prob[idx]
        vals.append(point_metrics(yy, (pp >= 0.5).astype(int), pp)[metric])
    v = np.asarray(vals)
    obs = point_metrics(y, (prob >= 0.5).astype(int), prob)[metric]
    return obs, float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def paired_test(y, prob_a, prob_b, idx_sets, metric="ROC-AUC") -> dict:
    """Paired cluster bootstrap: is model A actually better than model B?"""
    diffs = []
    for idx in idx_sets:
        yy = y[idx]
        if len(np.unique(yy)) < 2:
            continue
        a = point_metrics(yy, (prob_a[idx] >= 0.5).astype(int), prob_a[idx])[metric]
        b = point_metrics(yy, (prob_b[idx] >= 0.5).astype(int), prob_b[idx])[metric]
        diffs.append(a - b)
    d = np.asarray(diffs)
    obs = (point_metrics(y, (prob_a >= 0.5).astype(int), prob_a)[metric]
           - point_metrics(y, (prob_b >= 0.5).astype(int), prob_b)[metric])
    p = 2 * min((d <= 0).mean(), (d >= 0).mean())
    return {"diff": float(obs), "lo": float(np.percentile(d, 2.5)),
            "hi": float(np.percentile(d, 97.5)), "p": float(min(p, 1.0))}


# ------------------------------------------------------- calibration & decision curves
def ece(y, prob, bins=10) -> float:
    edges = np.quantile(prob, np.linspace(0, 1, bins + 1))
    edges = np.unique(edges)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (prob >= lo) & (prob <= hi if hi == edges[-1] else prob < hi)
        if m.sum() == 0:
            continue
        total += m.mean() * abs(y[m].mean() - prob[m].mean())
    return float(total)


def net_benefit(y, prob, thresholds) -> np.ndarray:
    n = len(y)
    nb = []
    for pt in thresholds:
        pred = prob >= pt
        tp = np.sum(pred & (y == 1))
        fp = np.sum(pred & (y == 0))
        nb.append(tp / n - (fp / n) * (pt / (1 - pt)))
    return np.asarray(nb)


def fig_calibration_dca(y, res: dict, path: Path) -> None:
    models = [m for m in res if m != "Majority class"]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4))

    ax = axes[0]
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
    for m in models:
        p = res[m]["prob"]
        edges = np.unique(np.quantile(p, np.linspace(0, 1, 9)))
        xs, ys = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            msk = (p >= lo) & (p < hi) if hi != edges[-1] else (p >= lo)
            if msk.sum() >= 20:
                xs.append(p[msk].mean())
                ys.append(y[msk].mean())
        ax.plot(xs, ys, "o-", lw=1.8, ms=4, color=MODEL_COLOURS[m],
                label=f"{m} (Brier {brier_score_loss(y, p):.3f})")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title("(a) Calibration under Protocol B", fontsize=10)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.25)

    ax = axes[1]
    ts = np.linspace(0.05, 0.85, 90)
    prev = y.mean()
    ax.plot(ts, prev - (1 - prev) * (ts / (1 - ts)), color="k", lw=1.2, ls="--",
            label="Screen everyone")
    ax.axhline(0, color="k", lw=1.2, label="Screen no one")
    for m in models:
        ax.plot(ts, net_benefit(y, res[m]["prob"], ts), lw=2, color=MODEL_COLOURS[m], label=m)
    ax.set_ylim(-0.12, prev + 0.05)
    ax.set_xlabel("Threshold probability (harm of a missed case ÷ harm of a false alarm)")
    ax.set_ylabel("Net benefit")
    ax.set_title("(b) Decision-curve analysis", fontsize=10)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.25)

    for ax in axes:
        ax.set_axisbelow(True)
    fig.suptitle("Clinical usefulness, not just discrimination", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_forest(summary: pd.DataFrame, path: Path) -> None:
    """Both scorings with bootstrap CIs. Overlapping intervals = no separable ranking."""
    s = summary.sort_values("AUC_pat")
    fig, ax = plt.subplots(figsize=(10, 5.2))
    yy = np.arange(len(s))

    ax.errorbar(s["AUC"], yy + 0.16,
                xerr=[s["AUC"] - s["AUC_lo"], s["AUC_hi"] - s["AUC"]],
                fmt="o", ms=6.5, lw=1.7, capsize=3.5,
                color=PALETTE["sim"], ecolor=PALETTE["sim"],
                label="Row-weighted (as usually reported)")
    ax.errorbar(s["AUC_pat"], yy - 0.16,
                xerr=[s["AUC_pat"] - s["AUC_pat_lo"], s["AUC_pat_hi"] - s["AUC_pat"]],
                fmt="D", ms=6, lw=1.9, capsize=3.5,
                color=PALETTE["obs"], ecolor=PALETTE["obs"],
                label="Pattern-weighted (one point per questionnaire)")
    for i, (_, r) in enumerate(s.iterrows()):
        ax.text(1.012, i - 0.16, f"{r['AUC_pat']:.3f} [{r['AUC_pat_lo']:.3f}–{r['AUC_pat_hi']:.3f}]",
                va="center", fontsize=8, color=PALETTE["obs"], family="monospace")
    ax.axvline(0.5, color="k", ls="--", lw=1)
    ax.set_yticks(yy, s["Model"], fontsize=9)
    ax.set_xlabel("ROC-AUC (95% bootstrap CI, resampling response patterns)")
    ax.set_xlim(0.38, 1.16)
    ax.legend(fontsize=8.5, loc="lower left")
    ax.set_title("Protocol B: overlapping intervals mean the ranking is not separable",
                 fontsize=11)
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------- main
def main() -> None:
    global ENCODING
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--encoding", default="onehot", choices=["onehot", "ordinal", "hybrid"])
    args = ap.parse_args()
    ENCODING = args.encoding

    prep = prepare(verbose=True)
    X, y, groups = prep["X"], prep["y"], prep["groups"]
    yv = y.to_numpy()
    rng = np.random.default_rng(SEED)

    results = {
        "A": run_repeated_cv(X, y, None, "A", args.repeats),
        "B": run_repeated_cv(X, y, groups, "B", args.repeats),
    }

    # Protocol C: one row per distinct questionnaire (majority label on the 1 tie).
    first = pd.Series(groups).drop_duplicates().index
    Xd, yd = X.loc[first].reset_index(drop=True), y.loc[first].reset_index(drop=True)
    results["C"] = run_repeated_cv(Xd, yd, None, "C", args.repeats)

    # ---- cluster-bootstrap CIs and paired tests on Protocol B
    print(f"\n{'=' * 78}\nCluster-bootstrap CIs and paired tests (Protocol B, "
          f"{N_BOOT} resamples of response patterns)\n{'=' * 78}")
    idx_sets = cluster_boot_indices(groups, N_BOOT, rng)

    n_pat = len(np.unique(groups))
    pat_sets = pattern_boot_indices(n_pat, N_BOOT, rng)

    rows = []
    for m, r in results["B"].items():
        auc, lo, hi = boot_ci(yv, r["prob"], idx_sets, "ROC-AUC")
        f1, f1lo, f1hi = boot_ci(yv, r["prob"], idx_sets, "F1")
        pm = point_metrics(yv, (r["prob"] >= 0.5).astype(int), r["prob"])
        # pattern-weighted: one point per distinct questionnaire
        py, pp = pattern_aggregate(yv, r["prob"], groups)
        pauc, plo, phi = boot_ci(py, pp, pat_sets, "ROC-AUC")
        pf1 = point_metrics(py, (pp >= 0.5).astype(int), pp)["F1"]
        rows.append({"Model": m, "AUC": auc, "AUC_lo": lo, "AUC_hi": hi,
                     "AUC_pat": pauc, "AUC_pat_lo": plo, "AUC_pat_hi": phi,
                     "dAUC_weighting": auc - pauc,
                     "F1": f1, "F1_lo": f1lo, "F1_hi": f1hi, "F1_pat": pf1,
                     "Accuracy": pm["Accuracy"], "Brier": pm["Brier"],
                     "ECE": ece(yv, r["prob"]), "MCC": pm["MCC"]})
    summary = pd.DataFrame(rows).sort_values("AUC_pat", ascending=False).reset_index(drop=True)
    print(summary.round(3).to_string(index=False))
    summary.to_csv(OUT / "eval_protocolB_summary.csv", index=False)

    # Paired tests on the pattern-weighted metric — the estimate we actually defend.
    ranked = summary["Model"].tolist()
    pairs, best = [], ranked[0]
    pat = {m: pattern_aggregate(yv, r["prob"], groups) for m, r in results["B"].items()}
    for other in ranked[1:]:
        t = paired_test(pat[best][0], pat[best][1], pat[other][1], pat_sets)
        t.update({"A": best, "B": other})
        pairs.append(t)
    ptab = pd.DataFrame(pairs)[["A", "B", "diff", "lo", "hi", "p"]]
    print(f"\nPaired ΔAUC vs best model ({best}):")
    print(ptab.round(4).to_string(index=False))
    ptab.to_csv(OUT / "eval_paired_tests.csv", index=False)

    # ---- protocol comparison across A / B / C
    comp = []
    for m in results["B"]:
        row = {"Model": m}
        for proto in ("A", "B", "C"):
            src = results[proto][m]
            yy = yd.to_numpy() if proto == "C" else yv
            row[f"AUC_{proto}"] = point_metrics(
                yy, (src["prob"] >= 0.5).astype(int), src["prob"])["ROC-AUC"]
        py, pp = pattern_aggregate(yv, results["B"][m]["prob"], groups)
        row["AUC_B_pattern"] = roc_auc_score(py, pp) if len(np.unique(py)) > 1 else 0.5
        row["leakage_drop"] = row["AUC_A"] - row["AUC_B"]
        row["weighting_drop"] = row["AUC_B"] - row["AUC_B_pattern"]
        row["total_drop"] = row["AUC_A"] - row["AUC_B_pattern"]
        comp.append(row)
    ctab = pd.DataFrame(comp)
    print("\nProtocol comparison (AUC) — the two-stage deflation:")
    print(ctab.round(3).to_string(index=False))
    ctab.to_csv(OUT / "eval_protocol_comparison.csv", index=False)

    fig_calibration_dca(yv, results["B"], OUT / "fig_eval_calibration_dca.png")
    fig_forest(summary, OUT / "fig_eval_forest.png")

    (OUT / "eval_statistics.json").write_text(json.dumps({
        "repeats": args.repeats, "encoding": ENCODING, "n_bootstrap": N_BOOT,
        "n_rows": int(len(X)), "n_patterns": int(len(np.unique(groups))),
        "summary_protocolB": summary.round(4).to_dict(orient="records"),
        "paired_tests": ptab.round(4).to_dict(orient="records"),
        "protocol_comparison": ctab.round(4).to_dict(orient="records"),
    }, indent=2), encoding="utf-8")
    np.savez_compressed(OUT / "eval_oof.npz", y_true=yv,
                        **{f"{p}::{m}": r["prob"] for p in "AB" for m, r in results[p].items()})
    print(f"\nArtefacts written to {OUT}")


if __name__ == "__main__":
    main()
