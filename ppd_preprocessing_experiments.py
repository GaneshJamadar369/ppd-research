"""
Advanced preprocessing, evaluated rather than assumed.

Each preprocessing decision is tested against alternatives under the leakage-controlled
protocol, because on this dataset a technique that looks better on rows can be worse
on the questionnaires that matter.

P1  Imputation recovery benchmark
      Cells are masked MCAR on the DISTINCT complete questionnaires and each method has
      to recover them. Masking on rows would let a method "recover" a cell by copying the
      identical duplicate row, which measures duplication, not imputation.
      Methods: mode, marginal sampling, KNN (Hamming), MICE (logistic), MICE (random
      forest), regularised iterative MCA imputation (Josse & Husson).

P2  Encoding benchmark (9 encodings x 2 models)
      one-hot, one-hot + severity score (current default), ordinal, thermometer,
      hybrid, weight-of-evidence, target encoding (cross-fitted), MCA components,
      one-hot + pairwise interactions.

P3  Training-strategy benchmark
      standard, inverse-pattern-frequency sample weights, deduplicated training folds,
      balanced class weights, dropping the two null items (Anxious, Appetite).

P2/P3 use StratifiedGroupKFold (5 folds x 5 repeats) grouped on the questionnaire,
fixed hyperparameters (identical across configurations, so differences are due to the
preprocessing), and pattern-weighted AUC as the primary metric. Configurations are
compared with the Nadeau-Bengio corrected resampled t-test, Holm-adjusted.

    python ppd_preprocessing_experiments.py
"""

from __future__ import annotations

import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 4))

import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler, TargetEncoder
from statsmodels.stats.multitest import multipletests

from ppd_preprocessing import (AGE_COL, ORD_SUFFIX, SYMPTOM_COLS, build_preprocessor, level_order,
                               prepare)

warnings.filterwarnings("ignore")

OUT = Path(__file__).resolve().parent / "results"
SEED = 11
ITEMS = [AGE_COL] + SYMPTOM_COLS
ORD = [c + ORD_SUFFIX for c in ITEMS]
K = [len(level_order(c)) for c in ITEMS]
NULL_ITEMS = ["Feeling anxious", "Overeating or loss of appetite"]
SHORT = dict(zip(ITEMS, ["Age", "Sad/tearful", "Irritable", "Sleep trouble", "Concentration",
                         "Appetite", "Anxious", "Guilt", "Bonding"]))
PAL = {"accent": "#3b6ea5", "pat": "#c1443c", "grey": "#6b7280", "ink": "#1f2937", "row": "#8fb8de"}


# ============================================================ P1 imputation recovery
def impute_mode(Z: np.ndarray, M: np.ndarray, rng) -> np.ndarray:
    out = Z.copy()
    for j in range(Z.shape[1]):
        obs = Z[~M[:, j], j]
        out[M[:, j], j] = np.bincount(obs, minlength=K[j]).argmax()
    return out


def impute_marginal(Z, M, rng):
    out = Z.copy()
    for j in range(Z.shape[1]):
        obs = Z[~M[:, j], j]
        p = np.bincount(obs, minlength=K[j]) / len(obs)
        out[M[:, j], j] = rng.choice(K[j], size=M[:, j].sum(), p=p)
    return out


def impute_knn_hamming(Z, M, rng, k=7):
    out = impute_mode(Z, M, rng)
    for i, j in zip(*np.nonzero(M)):
        other = np.arange(Z.shape[1]) != j
        both = (~M[:, other]) & (~M[i, other])
        dist = ((Z[:, other] != Z[i, other]) & both).sum(1) / np.maximum(both.sum(1), 1)
        cand = np.flatnonzero(~M[:, j])
        cand = cand[cand != i]
        nn = cand[np.argsort(dist[cand], kind="stable")[:k]]
        votes = np.bincount(Z[nn, j], minlength=K[j])
        out[i, j] = votes.argmax() if votes.max() > 0 else out[i, j]
    return out


def _onehot_codes(Z, skip):
    cols = []
    for j in range(Z.shape[1]):
        if j == skip:
            continue
        cols.append(np.eye(K[j])[Z[:, j]])
    return np.hstack(cols)


def impute_mice(Z, M, rng, learner="logistic", rounds=4):
    out = impute_mode(Z, M, rng)
    for _ in range(rounds):
        for j in range(Z.shape[1]):
            miss = M[:, j]
            if not miss.any():
                continue
            Xf = _onehot_codes(out, j)
            yj = out[~miss, j]
            if len(np.unique(yj)) < 2:
                continue
            if learner == "logistic":
                clf = LogisticRegression(max_iter=2000, C=1.0)
            else:
                clf = ExtraTreesClassifier(n_estimators=80, min_samples_leaf=2,
                                           random_state=int(rng.integers(1e9)), n_jobs=1)
            clf.fit(Xf[~miss], yj)
            out[miss, j] = clf.predict(Xf[miss])
    return out


def impute_iterative_mca(Z, M, rng, dims=2, iters=200, tol=1e-6):
    """Regularised iterative MCA imputation (Josse, Chavent, Liquet & Husson, 2012)."""
    n, Q = Z.shape
    offs = np.concatenate([[0], np.cumsum(K)[:-1]])
    J = sum(K)
    T = np.zeros((n, J))
    miss_mask = np.zeros((n, J), dtype=bool)
    for j in range(Q):
        T[np.arange(n), offs[j] + Z[:, j]] = 1.0
        block = slice(offs[j], offs[j] + K[j])
        T[M[:, j], block] = 0.0
        miss_mask[M[:, j], block] = True
    for j in range(Q):                                      # start at column proportions
        block = slice(offs[j], offs[j] + K[j])
        prop = T[~M[:, j], block].mean(0)
        T[np.ix_(M[:, j], np.arange(offs[j], offs[j] + K[j]))] = prop

    prev = np.inf
    for _ in range(iters):
        p = np.clip(T.mean(0), 1e-9, None)
        S = (T - p) / np.sqrt(Q * p)
        U, s, Vt = np.linalg.svd(S, full_matrices=False)
        sigma2 = (s[dims:] ** 2).mean()                     # noise level from discarded dims
        s_reg = np.clip((s[:dims] ** 2 - sigma2) / s[:dims], 0, None)
        S_hat = (U[:, :dims] * s_reg) @ Vt[:dims]
        T_hat = S_hat * np.sqrt(Q * p) + p
        change = np.abs(T_hat[miss_mask] - T[miss_mask]).max() if miss_mask.any() else 0
        T[miss_mask] = T_hat[miss_mask]
        if change < tol or abs(prev - change) < tol:
            break
        prev = change

    out = Z.copy()
    for j in range(Q):
        rows = np.flatnonzero(M[:, j])
        if len(rows):
            out[rows, j] = T[rows, offs[j]:offs[j] + K[j]].argmax(1)
    return out


IMPUTERS = {
    "Marginal sampling": impute_marginal,
    "Mode": impute_mode,
    "KNN (Hamming, k=7)": impute_knn_hamming,
    "MICE (logistic)": lambda Z, M, r: impute_mice(Z, M, r, "logistic"),
    "MICE (random forest)": lambda Z, M, r: impute_mice(Z, M, r, "forest"),
    "Iterative MCA (2 dims)": impute_iterative_mca,
}


def p1_imputation(prep: dict, reps: int = 30, rate: float = 0.10) -> tuple[pd.DataFrame, pd.DataFrame]:
    X, g = prep["X"], prep["groups"]
    first = pd.Series(g).drop_duplicates().index.to_numpy()
    codes = np.column_stack([
        X[c].map({v: i for i, v in enumerate(level_order(c))}).to_numpy() for c in ITEMS])[first]
    complete = ~np.isnan(codes.astype(float)).any(1)
    Z = codes[complete].astype(int)
    rng = np.random.default_rng(SEED)
    print(f"P1: {len(Z)} complete distinct questionnaires, masking {rate:.0%} of cells x {reps}",
          flush=True)

    rows, per_item = [], []
    for rep in range(reps):
        M = rng.random(Z.shape) < rate
        M[M.all(1)] = False
        for name, fn in IMPUTERS.items():
            imp = fn(Z, M, np.random.default_rng(rep))
            hit = imp[M] == Z[M]
            rows.append({"rep": rep, "Method": name, "Accuracy": hit.mean(), "Cells": int(M.sum())})
            for j in range(Z.shape[1]):
                if M[:, j].any():
                    per_item.append({"rep": rep, "Method": name, "Item": ITEMS[j],
                                     "Accuracy": (imp[M[:, j], j] == Z[M[:, j], j]).mean()})
    return pd.DataFrame(rows), pd.DataFrame(per_item)


# ======================================================== P2 custom fold-safe encoders
class WoEEncoder(BaseEstimator, TransformerMixin):
    """Weight of evidence per category, fitted on the training fold only."""

    def __init__(self, smoothing: float = 0.5):
        self.smoothing = smoothing

    def fit(self, X, y):
        X = np.asarray(X, dtype=object)
        y = np.asarray(y).astype(int)
        self.maps_ = []
        for j in range(X.shape[1]):
            cats = pd.unique(X[:, j])
            pos, neg = y.sum(), len(y) - y.sum()
            m = {}
            for c in cats:
                sel = X[:, j] == c
                p = (y[sel].sum() + self.smoothing) / (pos + self.smoothing * len(cats))
                q = ((1 - y[sel]).sum() + self.smoothing) / (neg + self.smoothing * len(cats))
                m[c] = np.log(p / q)
            self.maps_.append(m)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=object)
        return np.column_stack([[m.get(v, 0.0) for v in X[:, j]] for j, m in enumerate(self.maps_)])


class ThermometerEncoder(BaseEstimator, TransformerMixin):
    """Cumulative indicators 1[code >= k] — the ordinal analogue of one-hot."""

    def __init__(self, levels: tuple = tuple(K)):
        self.levels = levels

    def fit(self, X, y=None):
        self.n_features_in_ = np.asarray(X).shape[1]    # marks the encoder as fitted
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return np.hstack([(X[:, [j]] >= np.arange(1, k)).astype(float)
                          for j, k in enumerate(self.levels)])


class MCATransformer(BaseEstimator, TransformerMixin):
    """Row principal coordinates from an MCA fitted on the training fold's indicator matrix."""

    def __init__(self, n_dims: int = 8, n_items: int = len(ITEMS)):
        self.n_dims = n_dims
        self.n_items = n_items

    def fit(self, Z, y=None):
        Z = np.asarray(Z, dtype=float)
        P = Z / Z.sum()
        r, c = P.sum(1), np.clip(P.sum(0), 1e-12, None)
        S = (P - np.outer(r, c)) / np.sqrt(np.outer(r, c))
        _, _, Vt = np.linalg.svd(S, full_matrices=False)
        self.B_ = (Vt.T / np.sqrt(c)[:, None])[:, : self.n_dims]
        return self

    def transform(self, Z):
        return np.asarray(Z, dtype=float) @ self.B_ / self.n_items


def _cat_imputer():
    return SimpleImputer(strategy="most_frequent")


def encoders() -> dict[str, callable]:
    oh = lambda: OneHotEncoder(handle_unknown="ignore", sparse_output=False)  # noqa: E731
    return {
        "One-hot": lambda: ColumnTransformer([("c", Pipeline([("i", _cat_imputer()), ("e", oh())]), ITEMS)]),
        "One-hot + severity score (current)": lambda: build_preprocessor("onehot"),
        "Ordinal codes": lambda: ColumnTransformer([("o", SimpleImputer(strategy="median"), ORD)]),
        "Thermometer": lambda: ColumnTransformer(
            [("t", Pipeline([("i", _cat_imputer()), ("e", ThermometerEncoder())]), ORD)]),
        "Hybrid (one-hot + ordinal)": lambda: build_preprocessor("hybrid"),
        "Weight of evidence": lambda: ColumnTransformer(
            [("w", Pipeline([("i", _cat_imputer()), ("e", WoEEncoder())]), ITEMS)]),
        "Target encoding (cross-fitted)": lambda: ColumnTransformer(
            [("t", Pipeline([("i", _cat_imputer()),
                             ("e", TargetEncoder(target_type="binary", random_state=SEED))]), ITEMS)]),
        "MCA components (8)": lambda: ColumnTransformer(
            [("m", Pipeline([("i", _cat_imputer()), ("e", oh()), ("mca", MCATransformer(8))]), ITEMS)]),
        "One-hot + pairwise interactions": lambda: ColumnTransformer(
            [("c", Pipeline([("i", _cat_imputer()), ("e", oh()),
                             ("x", PolynomialFeatures(2, interaction_only=True, include_bias=False))]),
              ITEMS)]),
    }


def model(name: str, class_weight=None):
    if name == "LR":
        return [("scale", StandardScaler()),
                ("clf", LogisticRegression(C=0.5, max_iter=5000, class_weight=class_weight))]
    return [("clf", RandomForestClassifier(n_estimators=400, min_samples_leaf=3, random_state=SEED,
                                           n_jobs=-1,
                                           class_weight=("balanced_subsample" if class_weight else None)))]


def pattern_auc(y, p, g) -> float:
    df = pd.DataFrame({"g": g, "y": y, "p": p}).groupby("g").agg(y=("y", "mean"), p=("p", "mean"))
    yy = (df["y"] >= 0.5).astype(int)
    return float(roc_auc_score(yy, df["p"])) if yy.nunique() == 2 else np.nan


def run_config(X, y, g, make_pipe, strategy: str, splits) -> dict:
    fold_pat, fold_row = [], []
    oof_by_rep: dict[int, np.ndarray] = {}
    for rep, tr, te in splits:
        pipe = make_pipe()
        Xtr, ytr, gtr = X.iloc[tr], y[tr], g[tr]
        fit_kw = {}
        if strategy == "Inverse-frequency weights":
            cnt = pd.Series(gtr).map(pd.Series(gtr).value_counts()).to_numpy()
            fit_kw["clf__sample_weight"] = 1.0 / cnt
        elif strategy == "Deduplicated training":
            keep = ~pd.Series(gtr).duplicated().to_numpy()
            Xtr, ytr = Xtr[keep], ytr[keep]
        pipe.fit(Xtr, ytr, **fit_kw)
        p = pipe.predict_proba(X.iloc[te])[:, 1]
        fold_row.append(roc_auc_score(y[te], p))
        fold_pat.append(pattern_auc(y[te], p, g[te]))
        oof_by_rep.setdefault(rep, np.zeros(len(y)))[te] = p
    rep_pat = [pattern_auc(y, o, g) for o in oof_by_rep.values()]
    rep_row = [roc_auc_score(y, o) for o in oof_by_rep.values()]
    return {"fold_pat": np.array(fold_pat), "fold_row": np.array(fold_row),
            "rep_pat": np.array(rep_pat), "rep_row": np.array(rep_row)}


def nadeau_bengio(d: np.ndarray, test_train_ratio: float) -> tuple[float, float]:
    """Corrected resampled t-test (Nadeau & Bengio, 2003) for repeated CV differences."""
    d = d[~np.isnan(d)]
    k = len(d)
    var = d.var(ddof=1)
    if var == 0:
        return 0.0, 1.0
    t = d.mean() / np.sqrt((1 / k + test_train_ratio) * var)
    return float(t), float(2 * stats.t.sf(abs(t), k - 1))


def p2_p3(prep: dict, repeats: int = 5) -> pd.DataFrame:
    X, y, g = prep["X"], prep["y"].to_numpy(), prep["groups"]
    splits = []
    for rep in range(repeats):
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED + 97 * rep)
        splits += [(rep, tr, te) for tr, te in cv.split(X, y, g)]
    ratio = float(np.mean([len(te) / len(tr) for _, tr, te in splits]))

    configs = []
    for enc_name, enc in encoders().items():
        for m in ("LR", "RF"):
            configs.append(("Encoding", enc_name, m, "Standard",
                            (lambda enc=enc, m=m: Pipeline([("prep", enc())] + model(m)))))
    oh = encoders()["One-hot"]
    for strat in ("Inverse-frequency weights", "Deduplicated training", "Balanced class weights",
                  "Drop null items (Anxious, Appetite)"):
        for m in ("LR", "RF"):
            if strat == "Balanced class weights":
                mk = lambda m=m: Pipeline([("prep", oh())] + model(m, class_weight="balanced"))  # noqa: E731
            elif strat.startswith("Drop null"):
                keep = [c for c in ITEMS if c not in NULL_ITEMS]
                mk = lambda m=m, keep=keep: Pipeline([("prep", ColumnTransformer(  # noqa: E731
                    [("c", Pipeline([("i", _cat_imputer()),
                                     ("e", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]),
                      keep)]))] + model(m))
            else:
                mk = lambda m=m: Pipeline([("prep", oh())] + model(m))  # noqa: E731
            configs.append(("Strategy", strat, m, strat, mk))

    results = {}
    for i, (family, name, m, strategy, mk) in enumerate(configs, 1):
        run_strategy = strategy if family == "Strategy" else "Standard"
        if strategy in ("Balanced class weights",) or strategy.startswith("Drop null"):
            run_strategy = "Standard"
        results[(family, name, m)] = run_config(X, y, g, mk, run_strategy, splits)
        r = results[(family, name, m)]
        print(f"  [{i:2}/{len(configs)}] {m} | {name:<38} pattern AUC {np.nanmean(r['rep_pat']):.3f}"
              f"  row AUC {np.nanmean(r['rep_row']):.3f}", flush=True)

    rows = []
    for (family, name, m), r in results.items():
        base = results[("Encoding", "One-hot", m)]
        d = r["fold_pat"] - base["fold_pat"]
        t, p = nadeau_bengio(d, ratio) if name != "One-hot" or family != "Encoding" else (0.0, np.nan)
        rows.append({
            "Family": family, "Config": name, "Model": m,
            "Pattern AUC": float(np.nanmean(r["rep_pat"])), "Pattern AUC SD": float(np.nanstd(r["rep_pat"], ddof=1)),
            "Row AUC": float(np.nanmean(r["rep_row"])), "Row AUC SD": float(np.nanstd(r["rep_row"], ddof=1)),
            "Fold pattern AUC": float(np.nanmean(r["fold_pat"])),
            "Δ vs one-hot": float(np.nanmean(d)) if not (family == "Encoding" and name == "One-hot") else 0.0,
            "t (NB corrected)": t, "p (NB corrected)": p,
            "Row − pattern gap": float(np.nanmean(r["rep_row"]) - np.nanmean(r["rep_pat"])),
        })
    tab = pd.DataFrame(rows)
    for m in ("LR", "RF"):
        sel = (tab["Model"] == m) & tab["p (NB corrected)"].notna()
        tab.loc[sel, "q Holm"] = multipletests(tab.loc[sel, "p (NB corrected)"], method="holm")[1]
    tab.attrs["ratio"] = ratio
    return tab


# ======================================================================= figures
def _style(ax, axis="both"):
    ax.grid(axis=axis, alpha=0.25)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def fig_imputation(acc: pd.DataFrame, per_item: pd.DataFrame, path: Path) -> None:
    order = acc.groupby("Method")["Accuracy"].mean().sort_values().index.tolist()
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5), gridspec_kw={"width_ratios": [1, 1.15]})
    ax = axes[0]
    data = [acc.loc[acc["Method"] == m, "Accuracy"].to_numpy() for m in order]
    ax.boxplot(data, vert=False, widths=0.55, showfliers=False,
               medianprops={"color": PAL["ink"], "lw": 2})
    rng = np.random.default_rng(0)
    for i, v in enumerate(data, 1):
        ax.scatter(v, i + rng.uniform(-0.16, 0.16, len(v)), s=10, alpha=0.5,
                   color=PAL["pat"] if order[i - 1] == "Mode" else PAL["accent"])
    mode_mean = acc.loc[acc["Method"] == "Mode", "Accuracy"].mean()
    ax.axvline(mode_mean, color=PAL["pat"], ls="--", lw=1, label=f"Mode baseline {mode_mean:.3f}")
    ax.set_yticks(range(1, len(order) + 1), order)
    ax.set_xlabel("Share of masked cells recovered exactly")
    ax.set_title("(a) Imputation accuracy on distinct questionnaires (10% MCAR, 30 masks)",
                 fontsize=10, loc="left")
    ax.legend(fontsize=8.5, loc="lower right")
    _style(ax, "x")

    ax = axes[1]
    piv = per_item.groupby(["Item", "Method"])["Accuracy"].mean().unstack()
    piv = piv.loc[ITEMS, order]
    im = ax.imshow(piv.to_numpy(), cmap="Blues", vmin=0.2, vmax=0.9, aspect="auto")
    ax.set_xticks(range(len(order)), [o.replace(" (", "\n(") for o in order], fontsize=7.5)
    ax.set_yticks(range(len(ITEMS)), [SHORT[c] for c in ITEMS], fontsize=8.5)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.iat[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.5,
                    color="white" if v > 0.6 else PAL["ink"])
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    best = order[-1]
    gain = (piv[best] - piv["Mode"]).sort_values(ascending=False)
    ax.set_title(f"(b) Per-item recovery — largest gains of {best} over mode: "
                 + ", ".join(SHORT[i] for i in gain.index[:4]), fontsize=9.5, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_encoding(tab: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 7.2), sharey=True)
    enc = tab[tab["Family"] == "Encoding"]["Config"].drop_duplicates().tolist()
    strat = tab[tab["Family"] == "Strategy"]["Config"].drop_duplicates().tolist()
    order = enc + strat
    yy = np.arange(len(order))[::-1]
    for ax, m in zip(axes, ("LR", "RF")):
        sub = tab[tab["Model"] == m].set_index("Config").loc[order]
        base = sub.loc["One-hot", "Pattern AUC"]
        for yi, (name, r) in zip(yy, sub.iterrows()):
            sig = (r.get("q Holm", np.nan) < 0.05)
            colour = PAL["pat"] if sig else (PAL["accent"] if r["Family"] == "Encoding" else PAL["grey"])
            ax.errorbar(r["Pattern AUC"], yi, xerr=r["Pattern AUC SD"], fmt="o", color=colour,
                        ms=7 if sig else 5.5, capsize=3, lw=1.4)
            ax.scatter(r["Row AUC"], yi, marker="|", s=90, color=PAL["row"], zorder=2)
            label = f"{r['Pattern AUC']:.3f}"
            q = r.get("q Holm", np.nan)
            if not (r["Family"] == "Encoding" and name == "One-hot") and not pd.isna(q):
                label += f"   Δ {r['Δ vs one-hot']:+.3f}   q = {q:.2f}"
            ax.text(0.935, yi, label, ha="left", va="center", fontsize=7.8,
                    color=PAL["ink"] if sig else PAL["grey"], family="monospace")
        ax.axvline(base, color=PAL["ink"], ls="--", lw=1)
        ax.axvline(0.925, color=PAL["grey"], lw=0.6)
        ax.axhline(len(strat) - 0.5, color=PAL["grey"], lw=0.8)
        ax.set_title(f"{'Logistic regression' if m == 'LR' else 'Random forest'} — dots: pattern-weighted "
                     f"AUC ±SD (5 repeats); ticks: row-weighted", fontsize=9.5, loc="left")
        ax.set_xlabel("ROC-AUC under grouped CV")
        ax.set_xlim(0.55, 1.2)
        ax.set_xticks([0.6, 0.7, 0.8, 0.9])
        _style(ax, "x")
    axes[0].set_yticks(yy, order, fontsize=8.5)
    n_sig = int((tab["q Holm"] < 0.05).sum())
    verdict = ("red = significant" if n_sig else "no configuration differs significantly")
    fig.suptitle(f"Encodings (top) and training strategies (bottom) against one-hot — {verdict} "
                 f"(Nadeau–Bengio corrected t-test + Holm)", fontsize=11.5)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ========================================================================= main
def main() -> None:
    prep = prepare(verbose=False)

    cached = OUT / "prep_imputation_accuracy.csv", OUT / "prep_imputation_per_item.csv"
    if "--reuse-imputation" in sys.argv and all(p.exists() for p in cached):
        acc, per_item = pd.read_csv(cached[0]), pd.read_csv(cached[1])
        print("P1: reusing cached imputation benchmark", flush=True)
    else:
        acc, per_item = p1_imputation(prep)
        per_item.to_csv(cached[1], index=False)
    summ = acc.groupby("Method")["Accuracy"].agg(["mean", "std"]).sort_values("mean", ascending=False)
    mode = acc[acc["Method"] == "Mode"].set_index("rep")["Accuracy"]
    tests = []
    for mth in summ.index:
        if mth == "Mode":
            continue
        other = acc[acc["Method"] == mth].set_index("rep")["Accuracy"]
        w = stats.wilcoxon(other, mode, zero_method="zsplit")
        tests.append({"Method": mth, "Mean accuracy": summ.loc[mth, "mean"], "SD": summ.loc[mth, "std"],
                      "Δ vs mode": float((other - mode).mean()), "Wilcoxon p": float(w.pvalue)})
    tests = pd.DataFrame(tests)
    tests["q Holm"] = multipletests(tests["Wilcoxon p"], method="holm")[1]
    print("\nP1 imputation accuracy (mode = {:.3f}):".format(summ.loc["Mode", "mean"]))
    print(tests.round(4).to_string(index=False), flush=True)
    acc.to_csv(OUT / "prep_imputation_accuracy.csv", index=False)
    tests.to_csv(OUT / "prep_imputation_tests.csv", index=False)
    fig_imputation(acc, per_item, OUT / "fig_prep_imputation.png")

    print("\nP2/P3 encodings and strategies (grouped CV, 5 folds x 5 repeats)", flush=True)
    tab = p2_p3(prep)
    tab.to_csv(OUT / "prep_encoding_strategy_benchmark.csv", index=False)
    pd.set_option("display.width", 220)
    print(tab[["Family", "Config", "Model", "Pattern AUC", "Pattern AUC SD", "Row AUC",
               "Δ vs one-hot", "t (NB corrected)", "p (NB corrected)", "q Holm",
               "Row − pattern gap"]].round(4).to_string(index=False))
    fig_encoding(tab, OUT / "fig_prep_encoding.png")

    (OUT / "prep_summary.json").write_text(json.dumps({
        "imputation": tests.to_dict(orient="records"),
        "mode_accuracy": float(summ.loc["Mode", "mean"]),
        "benchmark": tab.to_dict(orient="records"),
        "nb_test_train_ratio": tab.attrs["ratio"],
    }, indent=2, default=float), encoding="utf-8")
    print(f"\nwritten to {OUT}")


if __name__ == "__main__":
    main()
