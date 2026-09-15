"""
Improved baselines for the PPD structured branch.

What limits the original baselines (ppd_evaluation.py, ppd_model_comparison.py):
  1. Hyperparameters are tuned on ROW-weighted ROC-AUC -- the metric that rewards
     memorising heavily repeated questionnaires -- while the paper reports the
     pattern-weighted AUC on distinct questionnaires.
  2. Grids are small and hand-picked (4-8 configurations per model).
  3. F1 / accuracy / recall are reported at a fixed 0.5 threshold.
  4. Three of the four models are high-variance learners for an effective n of ~248.

Improvements -- every choice is made inside the outer training fold, never on test data:
  * tuning objective = pattern-weighted AUC. The scorer groups the validation rows on
    their questionnaire (the feature vector itself), so it needs no group labels.
  * randomised search over wide, regularisation-heavy spaces, grouped inner CV
  * decision threshold chosen on grouped inner out-of-fold predictions (F1 on
    questionnaires)
  * learners suited to small categorical data: categorical naive Bayes, extra trees,
    gradient boosting with native categorical splits, Hamming-distance kNN, and
    elastic-net logistic regression with pairwise interactions
  * a PRE-SPECIFIED soft-voting ensemble (improved LR, improved RF, extra trees,
    categorical gradient boosting, categorical NB). Its members are fixed here, before
    any result is seen, and each is tuned inside the same training fold.

All models -- original and improved -- share identical outer splits (StratifiedGroupKFold,
5 folds x 3 repeats, grouped on the questionnaire), so every comparison is paired and
tested with the Nadeau-Bengio corrected resampled t-test, Holm-adjusted.

Per-model results are cached in results/baseline_cache/, so an interrupted run resumes.

    python ppd_improved_baselines.py            # run / resume
    python ppd_improved_baselines.py --fresh    # discard the cache
"""

from __future__ import annotations

import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 4))

import json
import pickle
import re
import shutil
import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import loguniform, randint, uniform
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, brier_score_loss, f1_score,
                             matthews_corrcoef, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV, StratifiedGroupKFold,
                                     cross_val_predict)
from sklearn.naive_bayes import CategoricalNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, PolynomialFeatures, StandardScaler
from sklearn.svm import SVC
from statsmodels.stats.multitest import multipletests

from ppd_preprocessing import AGE_COL, SYMPTOM_COLS, build_preprocessor, level_order, prepare

warnings.filterwarnings("ignore")

SEED = 2024
REPEATS, FOLDS, INNER = 3, 5, 3
N_BOOT = 2000
ITEMS = [AGE_COL] + SYMPTOM_COLS
LEVELS = [level_order(c) for c in ITEMS]
K = [len(lv) for lv in LEVELS]
OUT = Path(__file__).resolve().parent / "results"
CACHE = OUT / "baseline_cache"
CORE = ["LR", "SVM", "RF", "ANN"]
ENSEMBLE_MEMBERS = ["LR (improved)", "RF (improved)", "Extra Trees",
                    "Gradient Boosting (categorical)", "Categorical NB"]
PAL = {"orig": "#9aa5b1", "impr": "#3b6ea5", "new": "#4f8a5b", "ens": "#c1443c",
       "ink": "#1f2937", "grey": "#6b7280"}


# ============================================================ questionnaire-level scoring
def pattern_keys(X: pd.DataFrame) -> np.ndarray:
    return X[ITEMS].astype(str).agg("|".join, axis=1).to_numpy()


def to_patterns(keys: np.ndarray, y: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    df = pd.DataFrame({"k": keys, "y": y, "p": p}).groupby("k", sort=False).agg(
        y=("y", "mean"), p=("p", "mean"))
    return (df["y"].to_numpy() >= 0.5).astype(int), df["p"].to_numpy()


def pattern_auc_scorer(estimator, X, y) -> float:
    """Pattern-weighted AUC: each distinct questionnaire in the validation fold counts once."""
    yy, pp = to_patterns(pattern_keys(X), np.asarray(y), estimator.predict_proba(X)[:, 1])
    return float(roc_auc_score(yy, pp)) if len(np.unique(yy)) == 2 else 0.5


def choose_threshold(keys, y, p) -> float:
    """F1-maximising cut-off on questionnaire-level out-of-fold predictions."""
    yy, pp = to_patterns(keys, y, p)
    cand = np.unique(pp)
    if len(cand) > 150:
        cand = np.quantile(pp, np.linspace(0.02, 0.98, 97))
    scores = [f1_score(yy, (pp >= t).astype(int), zero_division=0) for t in cand]
    return float(cand[int(np.argmax(scores))])


# ======================================================================= preprocessors
def onehot() -> ColumnTransformer:
    return ColumnTransformer([("c", Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(categories=LEVELS, handle_unknown="ignore", sparse_output=False)),
    ]), ITEMS)])


def codes() -> ColumnTransformer:
    return ColumnTransformer([("c", Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OrdinalEncoder(categories=LEVELS)),
    ]), ITEMS)])


# ============================================================================== models
def model_specs() -> dict[str, dict]:
    S: dict[str, dict] = {}

    # ---- original baselines, reproduced exactly: small grids tuned on row-weighted AUC
    pre = lambda scale: build_preprocessor("onehot", scale=scale)  # noqa: E731
    S["LR (original)"] = dict(
        family="original", core="LR", search="grid", scoring="roc_auc",
        pipe=Pipeline([("prep", pre(True)), ("clf", LogisticRegression(max_iter=5000, random_state=SEED))]),
        space={"clf__C": [0.01, 0.1, 1.0, 10.0], "clf__class_weight": [None, "balanced"]})
    S["SVM (original)"] = dict(
        family="original", core="SVM", search="grid", scoring="roc_auc",
        pipe=Pipeline([("prep", pre(True)), ("clf", SVC(kernel="rbf", probability=True, random_state=SEED))]),
        space={"clf__C": [0.1, 1.0, 10.0], "clf__gamma": ["scale", 0.1]})
    S["RF (original)"] = dict(
        family="original", core="RF", search="grid", scoring="roc_auc",
        pipe=Pipeline([("prep", pre(False)),
                       ("clf", RandomForestClassifier(n_estimators=400, random_state=SEED, n_jobs=1))]),
        space={"clf__max_depth": [None, 6], "clf__min_samples_leaf": [1, 3, 10]})
    S["ANN (original)"] = dict(
        family="original", core="ANN", search="grid", scoring="roc_auc",
        pipe=Pipeline([("prep", pre(True)),
                       ("clf", MLPClassifier(max_iter=1500, early_stopping=True, n_iter_no_change=25,
                                             random_state=SEED))]),
        space={"clf__hidden_layer_sizes": [(32,), (64, 32)], "clf__alpha": [1e-4, 1e-2]})

    # ---- the same four models, improved
    S["LR (improved)"] = dict(
        family="improved", core="LR", search="random", n_iter=20,
        pipe=Pipeline([("prep", onehot()), ("scale", StandardScaler()),
                       ("clf", LogisticRegression(solver="saga", max_iter=5000, tol=1e-4,
                                                  random_state=SEED))]),
        space={"clf__C": loguniform(1e-3, 1e2), "clf__l1_ratio": uniform(0, 1),
               "clf__class_weight": [None, "balanced"]})
    S["SVM (improved)"] = dict(
        family="improved", core="SVM", search="random", n_iter=15,
        pipe=Pipeline([("prep", onehot()), ("scale", StandardScaler()),
                       ("clf", SVC(kernel="rbf", probability=True, random_state=SEED))]),
        space={"clf__C": loguniform(1e-2, 1e2), "clf__gamma": loguniform(1e-3, 1.0),
               "scale": [StandardScaler(), "passthrough"], "clf__class_weight": [None, "balanced"]})
    S["RF (improved)"] = dict(
        family="improved", core="RF", search="random", n_iter=20,
        pipe=Pipeline([("prep", onehot()),
                       ("clf", RandomForestClassifier(n_estimators=500, random_state=SEED, n_jobs=1))]),
        space={"clf__max_features": ["sqrt", "log2", 0.3, 0.5, 0.8],
               "clf__min_samples_leaf": randint(1, 26), "clf__max_depth": [None, 4, 6, 8, 12],
               "clf__criterion": ["gini", "entropy"], "clf__max_samples": [None, 0.5, 0.8],
               "clf__class_weight": [None, "balanced_subsample"]})
    S["ANN (improved)"] = dict(
        family="improved", core="ANN", search="random", n_iter=10,
        pipe=Pipeline([("prep", onehot()), ("scale", StandardScaler()),
                       ("clf", MLPClassifier(max_iter=2000, early_stopping=True, n_iter_no_change=20,
                                             random_state=SEED))]),
        space={"clf__hidden_layer_sizes": [(8,), (16,), (32,), (64,), (32, 16)],
               "clf__alpha": loguniform(1e-4, 10.0), "clf__learning_rate_init": loguniform(1e-3, 3e-2),
               "clf__activation": ["relu", "tanh"]})

    # ---- learners suited to small categorical data
    S["Categorical NB"] = dict(
        family="new", search="random", n_iter=15,
        pipe=Pipeline([("prep", codes()), ("clf", CategoricalNB(min_categories=K))]),
        space={"clf__alpha": loguniform(0.05, 20.0), "clf__fit_prior": [True, False]})
    S["Extra Trees"] = dict(
        family="new", search="random", n_iter=20,
        pipe=Pipeline([("prep", onehot()),
                       ("clf", ExtraTreesClassifier(n_estimators=500, random_state=SEED, n_jobs=1))]),
        space={"clf__max_features": ["sqrt", "log2", 0.3, 0.5, 0.8],
               "clf__min_samples_leaf": randint(1, 26), "clf__max_depth": [None, 4, 6, 8, 12],
               "clf__criterion": ["gini", "entropy"], "clf__class_weight": [None, "balanced"]})
    S["Gradient Boosting (categorical)"] = dict(
        family="new", search="random", n_iter=20,
        pipe=Pipeline([("prep", codes()),
                       ("clf", HistGradientBoostingClassifier(categorical_features=[True] * len(ITEMS),
                                                              early_stopping=False, random_state=SEED))]),
        space={"clf__learning_rate": loguniform(0.01, 0.3), "clf__max_iter": randint(50, 401),
               "clf__max_leaf_nodes": randint(4, 33), "clf__min_samples_leaf": randint(5, 61),
               "clf__l2_regularization": loguniform(1e-3, 10.0), "clf__max_depth": [None, 2, 3, 4],
               "clf__class_weight": [None, "balanced"]})
    S["kNN (Hamming)"] = dict(
        family="new", search="random", n_iter=15,
        pipe=Pipeline([("prep", onehot()),
                       ("clf", KNeighborsClassifier(metric="hamming", algorithm="brute"))]),
        space={"clf__n_neighbors": randint(5, 121), "clf__weights": ["uniform", "distance"]})
    S["LR + interactions (elastic net)"] = dict(
        family="new", search="random", n_iter=12,
        pipe=Pipeline([("prep", onehot()),
                       ("inter", PolynomialFeatures(2, interaction_only=True, include_bias=False)),
                       ("scale", StandardScaler()),
                       ("clf", LogisticRegression(solver="saga", max_iter=3000, tol=1e-3, random_state=SEED))]),
        space={"clf__C": loguniform(1e-3, 1e1), "clf__l1_ratio": uniform(0, 1)})
    return S


# ========================================================================= CV engine
def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def make_splits(X, y, g) -> list[tuple[int, int, np.ndarray, np.ndarray]]:
    splits = []
    for rep in range(REPEATS):
        cv = StratifiedGroupKFold(n_splits=FOLDS, shuffle=True, random_state=SEED + 101 * rep)
        for k, (tr, te) in enumerate(cv.split(X, y, g)):
            splits.append((rep, k, tr, te))
    return splits


def _jsonable_params(params: dict) -> dict:
    return {k: (v if isinstance(v, (int, float, str, bool, type(None), tuple, list)) else type(v).__name__)
            for k, v in params.items()}


def run_model(name, cfg, X, y, g, keys, splits) -> list[dict]:
    path = CACHE / f"{slug(name)}.pkl"
    if path.exists():
        with open(path, "rb") as fh:
            return pickle.load(fh)

    folds, t0 = [], time.time()
    for rep, k, tr, te in splits:
        rs = SEED + 13 * rep + k
        inner = StratifiedGroupKFold(n_splits=INNER, shuffle=True, random_state=rs)
        Xtr, ytr, gtr = X.iloc[tr], y[tr], g[tr]
        if cfg["search"] == "grid":
            search = GridSearchCV(cfg["pipe"], cfg["space"], scoring=cfg["scoring"], cv=inner,
                                  n_jobs=-1, refit=True)
        else:
            search = RandomizedSearchCV(cfg["pipe"], cfg["space"], n_iter=cfg["n_iter"],
                                        scoring=pattern_auc_scorer, cv=inner, n_jobs=-1,
                                        refit=True, random_state=rs)
        search.fit(Xtr, ytr, groups=gtr)
        best = search.best_estimator_
        prob = best.predict_proba(X.iloc[te])[:, 1]

        if cfg["family"] == "original":
            inner_oof, thr = None, 0.5                    # as originally reported
        else:
            inner_oof = cross_val_predict(clone(best), Xtr, ytr, groups=gtr, cv=inner,
                                          method="predict_proba", n_jobs=-1)[:, 1]
            thr = choose_threshold(keys[tr], ytr, inner_oof)
        folds.append({"rep": rep, "fold": k, "tr": tr, "te": te, "prob": prob, "thr": thr,
                      "inner_oof": inner_oof, "params": _jsonable_params(search.best_params_)})

    CACHE.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(folds, fh)
    print(f"    fitted in {time.time() - t0:.0f}s", flush=True)
    return folds


def build_ensemble(member_folds: list[list[dict]], keys, y) -> list[dict]:
    out = []
    for i in range(len(member_folds[0])):
        fs = [mf[i] for mf in member_folds]
        assert all(np.array_equal(f["te"], fs[0]["te"]) for f in fs), "members must share splits"
        inner = np.mean([f["inner_oof"] for f in fs], axis=0)
        out.append({"rep": fs[0]["rep"], "fold": fs[0]["fold"], "tr": fs[0]["tr"], "te": fs[0]["te"],
                    "prob": np.mean([f["prob"] for f in fs], axis=0),
                    "thr": choose_threshold(keys[fs[0]["tr"]], y[fs[0]["tr"]], inner),
                    "inner_oof": inner, "params": {"members": ENSEMBLE_MEMBERS}})
    return out


# ========================================================================= evaluation
def evaluate(folds: list[dict], keys, y) -> dict:
    reps = sorted({f["rep"] for f in folds})
    rows, fold_auc, pat_probs = [], [], []
    for r in reps:
        prob = np.full(len(y), np.nan)
        pred = np.zeros(len(y))
        for f in (f for f in folds if f["rep"] == r):
            prob[f["te"]] = f["prob"]
            pred[f["te"]] = (f["prob"] >= f["thr"]).astype(float)
            yy_f, pp_f = to_patterns(keys[f["te"]], y[f["te"]], f["prob"])
            fold_auc.append(roc_auc_score(yy_f, pp_f) if len(np.unique(yy_f)) == 2 else np.nan)
        yy, pp = to_patterns(keys, y, prob)
        _, pd_ = to_patterns(keys, y, pred)
        yhat = (pd_ >= 0.5).astype(int)
        tn = int(((yhat == 0) & (yy == 0)).sum())
        fp = int(((yhat == 1) & (yy == 0)).sum())
        rows.append({
            "AUC": roc_auc_score(yy, pp), "F1": f1_score(yy, yhat, zero_division=0),
            "Recall": recall_score(yy, yhat, zero_division=0),
            "Precision": precision_score(yy, yhat, zero_division=0),
            "Specificity": tn / (tn + fp) if tn + fp else 0.0,
            "Accuracy": accuracy_score(yy, yhat), "Balanced accuracy": balanced_accuracy_score(yy, yhat),
            "MCC": matthews_corrcoef(yy, yhat), "Brier": brier_score_loss(yy, np.clip(pp, 0, 1)),
            "Row AUC": roc_auc_score(y, prob),
        })
        pat_probs.append(pp)
    df = pd.DataFrame(rows)
    return {"reps": df, "fold_auc": np.array(fold_auc), "yy": yy, "pp_mean": np.mean(pat_probs, axis=0)}


def boot_auc_ci(yy, pp, rng) -> tuple[float, float]:
    n, vals = len(yy), []
    for _ in range(N_BOOT):
        i = rng.integers(0, n, n)
        if len(np.unique(yy[i])) == 2:
            vals.append(roc_auc_score(yy[i], pp[i]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def nadeau_bengio(d: np.ndarray, ratio: float) -> tuple[float, float]:
    d = d[~np.isnan(d)]
    var = d.var(ddof=1)
    if var == 0:
        return 0.0, 1.0
    t = d.mean() / np.sqrt((1 / len(d) + ratio) * var)
    return float(t), float(2 * stats.t.sf(abs(t), len(d) - 1))


# ============================================================================ figure
def figure(summary: pd.DataFrame, tests: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 7), gridspec_kw={"width_ratios": [1.35, 1]})
    s = summary.set_index("Model")

    ax = axes[0]
    order = ([f"{c} (improved)" for c in CORE] + ["Categorical NB", "Extra Trees",
             "Gradient Boosting (categorical)", "kNN (Hamming)", "LR + interactions (elastic net)",
             "Soft-vote ensemble"])
    labels = {f"{c} (improved)": c for c in CORE}
    yy = np.arange(len(order))[::-1]
    for yi, m in zip(yy, order):
        r = s.loc[m]
        colour = PAL["ens"] if m == "Soft-vote ensemble" else (PAL["impr"] if m.endswith("(improved)")
                                                                else PAL["new"])
        if m.endswith("(improved)"):
            o = s.loc[m.replace("improved", "original")]
            ax.errorbar(o["AUC"], yi, xerr=[[o["AUC"] - o["AUC lo"]], [o["AUC hi"] - o["AUC"]]],
                        fmt="o", mfc="white", color=PAL["orig"], ms=7, capsize=3, lw=1.2, zorder=2)
            ax.annotate("", xy=(r["AUC"], yi), xytext=(o["AUC"], yi),
                        arrowprops={"arrowstyle": "->", "color": PAL["grey"], "lw": 1.2})
        ax.errorbar(r["AUC"], yi, xerr=[[r["AUC"] - r["AUC lo"]], [r["AUC hi"] - r["AUC"]]], fmt="o",
                    color=colour, ms=8 if m == "Soft-vote ensemble" else 7, capsize=3, lw=1.6, zorder=3)
        t = tests[tests["Model"] == m]
        note = f"{r['AUC']:.3f}"
        if len(t):
            tr_ = t.iloc[0]
            note += f"   Δ {tr_['Δ AUC']:+.3f} vs {tr_['Reference'].split(' (')[0]} orig.   q = {tr_['q Holm']:.3f}"
        ax.text(0.935, yi, note, ha="left", va="center", fontsize=7.8, family="monospace",
                color=PAL["ink"] if len(t) and t.iloc[0]["q Holm"] < 0.05 else PAL["grey"])
    ax.axvline(s.loc["RF (original)", "AUC"], color=PAL["orig"], ls="--", lw=1)
    ax.axvline(0.925, color=PAL["grey"], lw=0.6)
    ax.set_yticks(yy, [labels.get(m, m) for m in order], fontsize=9)
    ax.set_xlim(0.5, 1.22)
    ax.set_xticks([0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9])
    ax.set_xlabel("Pattern-weighted ROC-AUC on distinct questionnaires (95% bootstrap CI)")
    ax.set_title("(a) Hollow = original baseline, filled = improved; dashed = best original (RF)",
                 fontsize=10, loc="left")
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)

    ax = axes[1]
    show = [f"{c} (original)" for c in CORE] + [f"{c} (improved)" for c in CORE] + ["Soft-vote ensemble"]
    show = [m for m in show if m in s.index]
    x = np.arange(len(show))
    w = 0.27
    for i, (metric, colour) in enumerate((("F1", PAL["impr"]), ("Recall", PAL["new"]),
                                          ("Balanced accuracy", PAL["ens"]))):
        ax.bar(x + (i - 1) * w, s.loc[show, metric], w, color=colour, label=metric, alpha=0.9)
    ax.set_xticks(x, [m.replace(" (", "\n(") for m in show], fontsize=7.6,
                 rotation=28, ha="right")
    ax.set_ylim(0, 1)
    ax.axvline(len(CORE) - 0.5, color=PAL["grey"], lw=0.8)
    ax.set_ylabel("Score on distinct questionnaires")
    ax.set_title("(b) Operating point — originals at 0.5, improved at a nested threshold",
                 fontsize=10, loc="left")
    ax.legend(fontsize=8.5, loc="upper left", ncol=3)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    for a in axes:
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    fig.suptitle("Improving the baselines — every choice nested inside the training folds, "
                 "scored on distinct questionnaires", fontsize=12)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.22)
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ============================================================================== main
def main() -> None:
    if "--fresh" in sys.argv and CACHE.exists():
        shutil.rmtree(CACHE)
    prep = prepare(verbose=False)
    X, y, g = prep["X"], prep["y"].to_numpy(), prep["groups"]
    keys = pattern_keys(X)
    splits = make_splits(X, y, g)
    ratio = float(np.mean([len(te) / len(tr) for _, _, tr, te in splits]))
    specs = model_specs()
    print(f"{len(X)} rows, {len(np.unique(g))} questionnaires, {len(splits)} outer folds, "
          f"{len(specs)} models + ensemble", flush=True)

    folds = {}
    for i, (name, cfg) in enumerate(specs.items(), 1):
        print(f"[{i:2}/{len(specs)}] {name}", flush=True)
        folds[name] = run_model(name, cfg, X, y, g, keys, splits)
    folds["Soft-vote ensemble"] = build_ensemble([folds[m] for m in ENSEMBLE_MEMBERS], keys, y)

    rng = np.random.default_rng(SEED)
    ev, rows = {}, []
    for name, fl in folds.items():
        e = evaluate(fl, keys, y)
        ev[name] = e
        lo, hi = boot_auc_ci(e["yy"], e["pp_mean"], rng)
        fam = specs[name]["family"] if name in specs else "ensemble"
        row = {"Model": name, "Family": fam, "AUC lo": lo, "AUC hi": hi}
        for col in e["reps"].columns:
            row[col] = float(e["reps"][col].mean())
            row[f"{col} SD"] = float(e["reps"][col].std(ddof=1))
        rows.append(row)
    summary = pd.DataFrame(rows).sort_values("AUC", ascending=False).reset_index(drop=True)

    tests = []
    for c in CORE:
        tests.append((f"{c} (improved)", f"{c} (original)"))
    for m in ["Categorical NB", "Extra Trees", "Gradient Boosting (categorical)", "kNN (Hamming)",
              "LR + interactions (elastic net)", "Soft-vote ensemble"]:
        tests.append((m, "RF (original)"))
    trows = []
    for m, ref in tests:
        d = ev[m]["fold_auc"] - ev[ref]["fold_auc"]
        t, p = nadeau_bengio(d, ratio)
        trows.append({"Model": m, "Reference": ref, "Δ AUC": float(np.nanmean(d)),
                      "Δ AUC (repeat-level)": float(ev[m]["reps"]["AUC"].mean() - ev[ref]["reps"]["AUC"].mean()),
                      "t (NB)": t, "p (NB)": p,
                      "Folds improved": f"{int(np.nansum(d > 0))}/{int(np.sum(~np.isnan(d)))}"})
    tests_df = pd.DataFrame(trows)
    tests_df["q Holm"] = multipletests(tests_df["p (NB)"], method="holm")[1]

    summary.to_csv(OUT / "baseline_improved_summary.csv", index=False)
    tests_df.to_csv(OUT / "baseline_improved_tests.csv", index=False)
    params = {}
    for name, fl in folds.items():
        counts = pd.Series([json.dumps(f["params"], sort_keys=True, default=str) for f in fl]).value_counts()
        params[name] = {"most_frequent": json.loads(counts.index[0]), "times_selected": int(counts.iloc[0]),
                        "per_fold": [f["params"] for f in fl]}
    (OUT / "baseline_improved_params.json").write_text(json.dumps(params, indent=2, default=str),
                                                        encoding="utf-8")
    figure(summary, tests_df, OUT / "fig_baseline_improvement.png")

    pd.set_option("display.width", 240)
    print("\n=== Questionnaire-level results (mean over 3 repeats; CI = bootstrap over questionnaires) ===")
    print(summary[["Model", "Family", "AUC", "AUC lo", "AUC hi", "AUC SD", "F1", "Recall", "Precision",
                   "Specificity", "Balanced accuracy", "MCC", "Brier", "Row AUC"]].round(3).to_string(index=False))
    print("\n=== Paired comparisons (Nadeau–Bengio corrected, Holm) ===")
    print(tests_df.round(4).to_string(index=False))
    print(f"\nwritten to {OUT}")


if __name__ == "__main__":
    main()
