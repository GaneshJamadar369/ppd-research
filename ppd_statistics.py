"""
Advanced statistical analysis of the PPD survey columns.

Every inferential statistic is computed in a way that respects the structure found in
the audit: 1,168 rows are only 248 distinct questionnaires, so rows are not independent
observations. Naive tests on rows are reported for comparison only; the defensible
versions are (i) design-effect corrected (Rao-Scott, Kish), (ii) computed on distinct
patterns, or (iii) cluster-robust / cluster-resampled on the response pattern.

Sections
  S1  Column profiles           counts, entropy, Gini-Simpson, imbalance, Wilson CIs
  S2  Univariate association    chi-square (naive / Rao-Scott / pattern-level), Fisher,
                                bias-corrected Cramer's V + cluster bootstrap CI, mutual
                                information + cluster permutation test, Theil's U,
                                Information Value, Holm & Benjamini-Hochberg
  S3  Monotonicity              Cochran-Armitage trend + departure-from-linearity test
  S4  Multivariable model       logistic regression, naive vs cluster-robust inference,
                                joint Wald tests, robust linearity contrasts
  S5  Inter-item dependence     bias-corrected V, polychoric, Theil's U, clustering, GVIF,
                                row-vs-pattern stability of the dependence structure
  S6  Scale structure           Cronbach / ordinal alpha, omega, item-rest, KMO,
                                Bartlett, Horn's parallel analysis
  S7  Correspondence analysis   MCA with Greenacre-adjusted inertia, supplementary target
  S8  Missingness               replication forensics, Fisher, Little's MCAR test
  S9  Duplication vs label      Kish effective n, prevalence shift, Mann-Whitney

    python ppd_statistics.py
"""

from __future__ import annotations

import os
import sys

# The Windows console defaults to cp1252 and cannot print symbols used in the digest.
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
import statsmodels.api as sm
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.optimize import minimize_scalar
from scipy.spatial.distance import squareform
from statsmodels.stats.multitest import multipletests

from ppd_preprocessing import AGE_COL, SCORE_COL, SYMPTOM_COLS, level_order, prepare

warnings.filterwarnings("ignore")

OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)
SEED = 2026
N_BOOT = 1000
N_PERM = 2000
BIG = 8.0

ITEMS = [AGE_COL] + SYMPTOM_COLS
SHORT = {
    AGE_COL: "Age",
    "Feeling sad or Tearful": "Sad/tearful",
    "Irritable towards baby & partner": "Irritable",
    "Trouble sleeping at night": "Sleep trouble",
    "Problems concentrating or making decision": "Concentration",
    "Overeating or loss of appetite": "Appetite",
    "Feeling anxious": "Anxious",
    "Feeling of guilt": "Guilt",
    "Problems of bonding with baby": "Bonding",
    "target": "TARGET",
}
PAL = {"row": "#8fb8de", "pat": "#c1443c", "accent": "#3b6ea5", "grey": "#6b7280",
       "ink": "#1f2937", "ok": "#4f8a5b"}


# =============================================================================== data
def load() -> dict:
    prep = prepare(verbose=False)
    X, y, g = prep["X"], prep["y"].to_numpy().astype(int), prep["groups"]
    K = [len(level_order(c)) for c in ITEMS]
    codes = np.full((len(X), len(ITEMS)), -1, dtype=int)
    for j, c in enumerate(ITEMS):
        m = {v: i for i, v in enumerate(level_order(c))}
        codes[:, j] = X[c].map(m).fillna(-1).astype(int).to_numpy()

    uniq, first, inv, sizes = np.unique(g, return_index=True, return_inverse=True,
                                        return_counts=True)
    n1 = np.bincount(inv, weights=y, minlength=len(uniq))
    lab = (n1 / sizes >= 0.5).astype(int)
    return {
        "X": X, "y": y, "g": g, "K": K, "codes": codes, "score": X[SCORE_COL].to_numpy(),
        "P_codes": codes[first], "P_size": sizes.astype(float), "P_n1": n1,
        "P_n0": sizes - n1, "P_lab": lab, "P_score": X[SCORE_COL].to_numpy()[first],
    }


# ============================================================================ helpers
def fmt_p(p: float) -> str:
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "—"
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def trim(t: np.ndarray) -> np.ndarray:
    t = t[t.sum(1) > 0]
    return t[:, t.sum(0) > 0]


def item_table(lv: np.ndarray, K: int, w0: np.ndarray, w1: np.ndarray) -> np.ndarray:
    v = lv >= 0
    return np.column_stack([np.bincount(lv[v], w0[v], K), np.bincount(lv[v], w1[v], K)])


def pair_table(a, b, Ka, Kb, w) -> np.ndarray:
    v = (a >= 0) & (b >= 0)
    return np.bincount(a[v] * Kb + b[v], w[v], Ka * Kb).reshape(Ka, Kb)


def chi2(t: np.ndarray) -> tuple[float, int, float]:
    t = trim(t)
    if min(t.shape) < 2:
        return 0.0, 0, 1.0
    s, p, df, _ = stats.chi2_contingency(t, correction=False)
    return float(s), int(df), float(p)


def cramers_v_bc(t: np.ndarray) -> float:
    """Bias-corrected Cramer's V (Bergsma 2013)."""
    t = trim(t)
    n = t.sum()
    r, k = t.shape
    if min(r, k) < 2 or n < 2:
        return 0.0
    phi2 = chi2(t)[0] / n
    phi2c = max(0.0, phi2 - (k - 1) * (r - 1) / (n - 1))
    rc, kc = r - (r - 1) ** 2 / (n - 1), k - (k - 1) ** 2 / (n - 1)
    d = min(rc - 1, kc - 1)
    return float(np.sqrt(phi2c / d)) if d > 0 else 0.0


def entropy_bits(counts: np.ndarray) -> float:
    p = counts[counts > 0] / counts.sum()
    return float(-(p * np.log2(p)).sum())


def mutual_info_bits(t: np.ndarray) -> float:
    n = t.sum()
    p = t / n
    e = np.outer(p.sum(1), p.sum(0))
    nz = p > 0
    return float((p[nz] * np.log2(p[nz] / e[nz])).sum())


def wilson(x: float, n: float, z: float = 1.96) -> tuple[float, float, float]:
    if n == 0:
        return np.nan, np.nan, np.nan
    p = x / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, c - h, c + h


def woe_iv(t: np.ndarray) -> tuple[np.ndarray, float]:
    """Weight of evidence per level (ln %pos / %neg; >0 = higher risk), with 0.5 smoothing."""
    neg, pos = t[:, 0] + 0.5, t[:, 1] + 0.5
    dn, dp = neg / neg.sum(), pos / pos.sum()
    woe = np.log(dp / dn)
    return woe, float(((dp - dn) * woe).sum())


def odds_ratio(t: np.ndarray, i: int, ref: int = 0) -> tuple[float, float, float]:
    a, b, c, d = t[i, 1], t[i, 0], t[ref, 1], t[ref, 0]
    if min(a, b, c, d) == 0:
        a, b, c, d = a + 0.5, b + 0.5, c + 0.5, d + 0.5
    lor, se = np.log(a * d / (b * c)), np.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    return float(np.exp(lor)), float(np.exp(lor - 1.96 * se)), float(np.exp(lor + 1.96 * se))


def cochran_armitage(t: np.ndarray) -> dict:
    """Trend test with equally spaced scores, and the exact partition of Pearson's X^2
    into a linear-trend component (1 df) and a departure-from-linearity component."""
    n_k, x_k = t.sum(1), t[:, 1]
    N, pbar = n_k.sum(), x_k.sum() / n_k.sum()
    s = np.arange(len(n_k), dtype=float)
    T = (s * (x_k - n_k * pbar)).sum()
    V = pbar * (1 - pbar) * ((n_k * s ** 2).sum() - (n_k * s).sum() ** 2 / N)
    z = T / np.sqrt(V)
    total = chi2(t)[0]
    dev, df = max(total - z * z, 0.0), len(n_k) - 2
    return {"z": float(z), "p_trend": float(2 * stats.norm.sf(abs(z))),
            "chi2_total": float(total), "chi2_trend": float(z * z),
            "chi2_dev": float(dev), "df_dev": int(df),
            "p_dev": float(stats.chi2.sf(dev, df)) if df > 0 else np.nan,
            "trend_share": float(z * z / total) if total > 0 else np.nan}


# ======================================================================= S1 profiles
def s1_profiles(D: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, levels = [], []
    y, codes = D["y"], D["codes"]
    for j, c in enumerate(ITEMS):
        lv, K = codes[:, j], D["K"][j]
        valid = lv >= 0
        counts = np.bincount(lv[valid], minlength=K).astype(float)
        pat_counts = np.bincount(D["P_codes"][:, j][D["P_codes"][:, j] >= 0], minlength=K)
        H = entropy_bits(counts)
        names = level_order(c)
        mode = int(np.argmax(counts))
        cum = np.cumsum(counts) / counts.sum()
        rows.append({
            "Item": c, "Short": SHORT[c], "Levels": K, "Valid n": int(valid.sum()),
            "Missing n": int((~valid).sum()),
            "Missing %": 100 * (~valid).mean(),
            "Mode": names[mode], "Mode share": counts[mode] / counts.sum(),
            "Median level": names[int(np.searchsorted(cum, 0.5))],
            "Entropy (bits)": H, "Norm. entropy": H / np.log2(K),
            "Gini-Simpson": 1 - ((counts / counts.sum()) ** 2).sum(),
            "Imbalance (max/min)": counts.max() / max(counts.min(), 1),
        })
        for k, name in enumerate(names):
            x = y[valid & (lv == k)].sum()
            n = counts[k]
            p, lo, hi = wilson(x, n)
            pm = D["P_codes"][:, j] == k
            pp, plo, phi = wilson(D["P_lab"][pm].sum(), pm.sum())
            levels.append({
                "Item": c, "Short": SHORT[c], "Level": name, "Code": k,
                "Rows": int(n), "Row %": 100 * n / counts.sum(),
                "Patterns": int(pat_counts[k]),
                "Target rate": p, "Rate lo": lo, "Rate hi": hi,
                "Pattern rate": pp, "Pattern lo": plo, "Pattern hi": phi,
            })
    return pd.DataFrame(rows), pd.DataFrame(levels)


# ==================================================================== S2 association
def s2_association(D: dict, rng) -> pd.DataFrame:
    P, sizes, n0, n1, lab = D["P_codes"], D["P_size"], D["P_n0"], D["P_n1"], D["P_lab"]
    N = sizes.sum()
    deff = float((sizes ** 2).sum() / N)          # Kish design effect, ICC = 1
    Hy = entropy_bits(np.array([n0.sum(), n1.sum()]))
    Pn = len(sizes)

    boot_w = [np.bincount(rng.integers(0, Pn, Pn), minlength=Pn) for _ in range(N_BOOT)]
    perm_lab = [rng.permutation(lab) for _ in range(N_PERM)]

    rows = []
    for j, c in enumerate(ITEMS):
        lv, K = P[:, j], D["K"][j]
        t_row = item_table(lv, K, n0, n1)
        t_pat = item_table(lv, K, 1.0 - lab, lab.astype(float))

        x2, df, p = chi2(t_row)
        x2_rs = x2 / deff
        x2_p, _, p_pat = chi2(t_pat)
        fisher = (stats.fisher_exact(trim(t_pat))[1] if trim(t_pat).shape == (2, 2)
                  else np.nan)

        v = cramers_v_bc(t_row)
        vb = [cramers_v_bc(item_table(lv, K, w * n0, w * n1)) for w in boot_w]
        v_pat = cramers_v_bc(t_pat)

        t_lab = item_table(lv, K, sizes * (1 - lab), sizes * lab)
        mi = mutual_info_bits(t_row)
        mi_lab = mutual_info_bits(t_lab)
        mi_null = np.array([mutual_info_bits(item_table(lv, K, sizes * (1 - pl), sizes * pl))
                            for pl in perm_lab])
        p_perm = (1 + (mi_null >= mi_lab - 1e-12).sum()) / (1 + N_PERM)

        _, iv = woe_iv(t_row)
        _, iv_pat = woe_iv(t_pat)
        rows.append({
            "Item": c, "Short": SHORT[c], "K": K,
            "Chi2 (rows)": x2, "df": df, "p (naive rows)": p,
            "Chi2 Rao-Scott": x2_rs, "p Rao-Scott": float(stats.chi2.sf(x2_rs, df)),
            "Chi2 (patterns)": x2_p, "p (patterns)": p_pat, "Fisher p (patterns)": fisher,
            "V_bc": v, "V_lo": float(np.percentile(vb, 2.5)),
            "V_hi": float(np.percentile(vb, 97.5)), "V_bc (patterns)": v_pat,
            "MI (bits)": mi, "U(target|item)": mi / Hy, "p perm (cluster)": p_perm,
            "IV": iv, "IV (patterns)": iv_pat,
        })
    df_ = pd.DataFrame(rows)
    df_["q Holm (Rao-Scott)"] = multipletests(df_["p Rao-Scott"], method="holm")[1]
    df_["q BH (patterns)"] = multipletests(df_["p (patterns)"], method="fdr_bh")[1]
    df_["q Holm (perm)"] = multipletests(df_["p perm (cluster)"], method="holm")[1]
    df_.attrs["deff"] = deff
    df_.attrs["kish_n"] = float(N / deff)
    return df_


def s2b_composite(D: dict) -> dict:
    """The engineered severity score as a single continuous predictor."""
    from sklearn.metrics import roc_auc_score

    y, s = D["y"], D["score"]
    ok = ~np.isnan(s)
    ps, pl = D["P_score"], D["P_lab"]
    pok = ~np.isnan(ps)
    mw = stats.mannwhitneyu(ps[pok & (pl == 1)], ps[pok & (pl == 0)], alternative="two-sided")
    rho, prho = stats.spearmanr(ps[pok], pl[pok])
    return {
        "AUC rows": float(roc_auc_score(y[ok], s[ok])),
        "AUC patterns": float(roc_auc_score(pl[pok], ps[pok])),
        "Mann-Whitney U (patterns)": float(mw.statistic), "MW p (patterns)": float(mw.pvalue),
        "Spearman rho (patterns)": float(rho), "Spearman p (patterns)": float(prho),
        "Mean score pos (patterns)": float(np.nanmean(ps[pl == 1])),
        "Mean score neg (patterns)": float(np.nanmean(ps[pl == 0])),
    }


# ==================================================================== S3 monotonicity
def s3_trend(D: dict) -> pd.DataFrame:
    rows = []
    for j, c in enumerate(ITEMS):
        K = D["K"][j]
        if K < 3:
            continue
        lv = D["P_codes"][:, j]
        for unit, (w0, w1) in {"rows": (D["P_n0"], D["P_n1"]),
                               "patterns": (1.0 - D["P_lab"], D["P_lab"].astype(float))}.items():
            ca = cochran_armitage(item_table(lv, K, w0, w1))
            rows.append({"Item": c, "Short": SHORT[c], "Unit": unit, "K": K, **ca})
    t = pd.DataFrame(rows)
    for unit in ("rows", "patterns"):
        m = t["Unit"] == unit
        t.loc[m, "q Holm (dev)"] = multipletests(t.loc[m, "p_dev"], method="holm")[1]
    return t


# ============================================================= S4 logistic regression
def design(codes: np.ndarray, K: list[int], linear: int | None = None):
    cols, names, blocks = [np.ones(len(codes))], ["Intercept"], {}
    for j, c in enumerate(ITEMS):
        lvls = level_order(c)
        if j == linear:
            cols.append(codes[:, j].astype(float))
            names.append(f"{SHORT[c]} (linear)")
            blocks[j] = [len(names) - 1]
            continue
        idx = []
        for k in range(1, K[j]):
            cols.append((codes[:, j] == k).astype(float))
            names.append(f"{SHORT[c]}: {lvls[k]} vs {lvls[0]}")
            idx.append(len(names) - 1)
        blocks[j] = idx
    return np.column_stack(cols), names, blocks


def s4_logistic(D: dict) -> dict:
    cc = (D["codes"] >= 0).all(axis=1)
    codes, y, g = D["codes"][cc], D["y"][cc], D["g"][cc]
    Xd, names, blocks = design(codes, D["K"])
    fam = sm.families.Binomial()
    naive = sm.GLM(y, Xd, family=fam).fit()
    robust = sm.GLM(y, Xd, family=fam).fit(cov_type="cluster", cov_kwds={"groups": g})
    null = sm.GLM(y, np.ones((len(y), 1)), family=fam).fit()

    ci_n, ci_r = naive.conf_int(), robust.conf_int()
    coef = pd.DataFrame({
        "Term": names, "OR": np.exp(naive.params),
        "CI naive lo": np.exp(ci_n[:, 0]), "CI naive hi": np.exp(ci_n[:, 1]),
        "CI robust lo": np.exp(ci_r[:, 0]), "CI robust hi": np.exp(ci_r[:, 1]),
        "p naive": naive.pvalues, "p robust": robust.pvalues,
        "SE inflation": robust.bse / naive.bse,
    })

    terms = []
    for j, c in enumerate(ITEMS):
        keep = [i for i in range(Xd.shape[1]) if i not in blocks[j]]
        reduced = sm.GLM(y, Xd[:, keep], family=fam).fit()
        lr = 2 * (naive.llf - reduced.llf)
        df = len(blocks[j])
        R = np.zeros((df, Xd.shape[1]))
        for r_, col in enumerate(blocks[j]):
            R[r_, col] = 1
        wald = robust.wald_test(R, scalar=True, use_f=False)
        row = {"Item": c, "Short": SHORT[c], "df": df, "LR chi2 (naive)": lr,
               "p LR (naive)": float(stats.chi2.sf(lr, df)),
               "Wald chi2 (robust)": float(wald.statistic),
               "p Wald (robust)": float(wald.pvalue)}
        if D["K"][j] >= 3:
            b = blocks[j]
            L = np.zeros((len(b) - 1, Xd.shape[1]))
            for r_, k in enumerate(range(2, len(b) + 1)):     # beta_k = k * beta_1
                L[r_, b[k - 1]] = 1
                L[r_, b[0]] = -k
            lin = robust.wald_test(L, scalar=True, use_f=False)
            Xl, _, _ = design(codes, D["K"], linear=j)
            lin_fit = sm.GLM(y, Xl, family=fam).fit()
            lr_lin = 2 * (naive.llf - lin_fit.llf)
            row.update({"Linearity LR (naive)": lr_lin,
                        "p linearity LR (naive)": float(stats.chi2.sf(lr_lin, len(b) - 1)),
                        "Linearity Wald (robust)": float(lin.statistic),
                        "p linearity (robust)": float(lin.pvalue)})
        terms.append(row)
    terms = pd.DataFrame(terms)
    terms["q Holm robust"] = multipletests(terms["p Wald (robust)"], method="holm")[1]
    if "p linearity (robust)" in terms:
        m = terms["p linearity (robust)"].notna()
        terms.loc[m, "q Holm linearity"] = multipletests(
            terms.loc[m, "p linearity (robust)"], method="holm")[1]

    # sensitivity: one row per distinct questionnaire
    pc = (D["P_codes"] >= 0).all(axis=1)
    Xp, _, _ = design(D["P_codes"][pc], D["K"])
    pat_fit = sm.GLM(D["P_lab"][pc], Xp, family=fam).fit()
    coef["OR (patterns)"] = np.exp(pat_fit.params)
    coef["p (patterns)"] = pat_fit.pvalues
    _, _, pblocks = design(D["P_codes"][pc], D["K"])
    pat_lr = []
    for j in range(len(ITEMS)):
        keep = [i for i in range(Xp.shape[1]) if i not in pblocks[j]]
        red = sm.GLM(D["P_lab"][pc], Xp[:, keep], family=fam).fit()
        lr_ = 2 * (pat_fit.llf - red.llf)
        pat_lr.append(float(stats.chi2.sf(lr_, len(pblocks[j]))))
    terms["p LR (patterns)"] = pat_lr
    terms["q Holm LR (patterns)"] = multipletests(pat_lr, method="holm")[1]

    return {
        "coef": coef, "terms": terms, "n": int(cc.sum()), "n_clusters": int(len(np.unique(g))),
        "mcfadden_r2": float(1 - naive.llf / null.llf),
        "lr_global": float(2 * (naive.llf - null.llf)), "df_global": int(Xd.shape[1] - 1),
        "median_se_inflation": float(np.median(coef["SE inflation"][1:])),
        "mcfadden_r2_patterns": float(1 - pat_fit.llf / sm.GLM(
            D["P_lab"][pc], np.ones((pc.sum(), 1)), family=fam).fit().llf),
    }


# ================================================================ S5 dependence
def thresholds(margin: np.ndarray) -> np.ndarray:
    cp = np.cumsum(margin)[:-1] / margin.sum()
    return np.concatenate([[-BIG], stats.norm.ppf(np.clip(cp, 1e-6, 1 - 1e-6)), [BIG]])


def polychoric(t: np.ndarray) -> float:
    """Two-step ML polychoric (tetrachoric when both variables are binary)."""
    t = trim(t)
    if min(t.shape) < 2:
        return np.nan
    a, b = thresholds(t.sum(1)), thresholds(t.sum(0))
    A, B = np.meshgrid(a, b, indexing="ij")
    pts = np.column_stack([A.ravel(), B.ravel()])

    def nll(r):
        cdf = stats.multivariate_normal(mean=[0, 0], cov=[[1, r], [r, 1]]).cdf(pts)
        cdf = np.asarray(cdf).reshape(A.shape)
        pi = cdf[1:, 1:] - cdf[:-1, 1:] - cdf[1:, :-1] + cdf[:-1, :-1]
        return -(t * np.log(np.clip(pi, 1e-12, None))).sum()

    return float(minimize_scalar(nll, bounds=(-0.995, 0.995), method="bounded",
                                 options={"xatol": 1e-4}).x)


def nearest_pd_corr(R: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    w, v = np.linalg.eigh((R + R.T) / 2)
    R2 = v @ np.diag(np.clip(w, eps, None)) @ v.T
    d = np.sqrt(np.diag(R2))
    return R2 / np.outer(d, d)


def s5_dependence(D: dict) -> dict:
    P, sizes, lab = D["P_codes"], D["P_size"], D["P_lab"]
    varsets = ITEMS + ["target"]
    Kall = D["K"] + [2]
    Pall = np.column_stack([P, lab])
    n = len(varsets)

    V = np.full((n, n), np.nan)
    Vp = np.full((n, n), np.nan)
    U = np.full((n, n), np.nan)
    Rpoly = np.eye(n)
    p_pairs_pat, p_pairs_rs, pairs = [], [], []
    deff = float((sizes ** 2).sum() / sizes.sum())
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            t = pair_table(Pall[:, i], Pall[:, j], Kall[i], Kall[j], sizes)
            tp = pair_table(Pall[:, i], Pall[:, j], Kall[i], Kall[j], np.ones_like(sizes))
            V[i, j] = cramers_v_bc(t)
            Vp[i, j] = cramers_v_bc(tp)
            U[i, j] = mutual_info_bits(trim(t)) / entropy_bits(trim(t).sum(1))
            if i < j:
                Rpoly[i, j] = Rpoly[j, i] = polychoric(t)
                if i < len(ITEMS) and j < len(ITEMS):
                    x2, df, _ = chi2(t)
                    p_pairs_rs.append(float(stats.chi2.sf(x2 / deff, df)))
                    p_pairs_pat.append(chi2(tp)[2])
                    pairs.append((SHORT[varsets[i]], SHORT[varsets[j]]))

    iu = np.triu_indices(n, 1)
    stability = float(np.corrcoef(V[iu], Vp[iu])[0, 1])

    q_pat = multipletests(p_pairs_pat, method="holm")[1]
    q_rs = multipletests(p_pairs_rs, method="holm")[1]
    pair_tab = pd.DataFrame({"Pair": [f"{a} × {b}" for a, b in pairs],
                             "p Rao-Scott": p_pairs_rs, "q Holm (RS)": q_rs,
                             "p patterns": p_pairs_pat, "q Holm (patterns)": q_pat})

    # Spearman on ordinal codes (row-level, complete cases) for comparison with polychoric
    cc = (D["codes"] >= 0).all(axis=1)
    sp = stats.spearmanr(np.column_stack([D["codes"][cc], D["y"][cc]]))[0]

    # GVIF on the one-hot design (drop-first), complete cases
    Xd, names, blocks = design(D["codes"][cc], D["K"])
    Rd = np.corrcoef(Xd[:, 1:], rowvar=False)
    det_all = np.linalg.slogdet(Rd)
    gvif = []
    for j, c in enumerate(ITEMS):
        b = [k - 1 for k in blocks[j]]
        rest = [k for k in range(Rd.shape[0]) if k not in b]
        lg = (np.linalg.slogdet(Rd[np.ix_(b, b)])[1] + np.linalg.slogdet(Rd[np.ix_(rest, rest)])[1]
              - det_all[1])
        g_ = float(np.exp(lg))
        gvif.append({"Item": c, "Short": SHORT[c], "df": len(b), "GVIF": g_,
                     "GVIF^(1/2df)": g_ ** (1 / (2 * len(b)))})

    return {"labels": [SHORT[v] for v in varsets], "V": V, "V_patterns": Vp, "U": U,
            "polychoric": Rpoly, "spearman": sp, "stability_r": stability,
            "pairs": pair_tab, "gvif": pd.DataFrame(gvif)}


# ============================================================ S6 scale structure
def cronbach_alpha(M: np.ndarray) -> float:
    k = M.shape[1]
    return float(k / (k - 1) * (1 - M.var(0, ddof=1).sum() / M.sum(1).var(ddof=1)))


def paf_one_factor(R: np.ndarray, iters: int = 500) -> np.ndarray:
    h2 = np.clip(1 - 1 / np.diag(np.linalg.inv(R)), 0.01, 0.99)
    lam = np.zeros(len(R))
    for _ in range(iters):
        Rr = R.copy()
        np.fill_diagonal(Rr, h2)
        w, v = np.linalg.eigh(Rr)
        lam = v[:, -1] * np.sqrt(max(w[-1], 0))
        new = np.clip(lam ** 2, 0.001, 0.999)
        if np.max(np.abs(new - h2)) < 1e-7:
            break
        h2 = new
    return lam * np.sign(lam.sum())


def kmo(R: np.ndarray) -> tuple[float, np.ndarray]:
    Ri = np.linalg.inv(R)
    Pm = -Ri / np.sqrt(np.outer(np.diag(Ri), np.diag(Ri)))
    np.fill_diagonal(Pm, 0)
    R0 = R.copy()
    np.fill_diagonal(R0, 0)
    per = (R0 ** 2).sum(0) / ((R0 ** 2).sum(0) + (Pm ** 2).sum(0))
    return float((R0 ** 2).sum() / ((R0 ** 2).sum() + (Pm ** 2).sum())), per


def s6_scale(D: dict, dep: dict, rng) -> dict:
    sym_idx = list(range(1, len(ITEMS)))            # 8 symptom items, exclude age
    names = [SHORT[ITEMS[j]] for j in sym_idx]

    out = {}
    for unit, codes in (("rows", D["codes"]), ("patterns", D["P_codes"])):
        M = codes[:, sym_idx]
        M = M[(M >= 0).all(axis=1)].astype(float)
        tot = M.sum(1)
        item_rest = [float(np.corrcoef(M[:, i], tot - M[:, i])[0, 1]) for i in range(M.shape[1])]
        a_del = [cronbach_alpha(np.delete(M, i, axis=1)) for i in range(M.shape[1])]
        out[unit] = {"alpha": cronbach_alpha(M), "item_rest": item_rest, "alpha_if_deleted": a_del,
                     "n": len(M)}

    R = nearest_pd_corr(dep["polychoric"][np.ix_(sym_idx, sym_idx)])
    k = len(sym_idx)
    rbar = (R.sum() - k) / (k * (k - 1))
    lam = paf_one_factor(R)
    omega = float(lam.sum() ** 2 / (lam.sum() ** 2 + (1 - lam ** 2).sum()))
    kmo_all, kmo_item = kmo(R)
    n_b = len(D["P_codes"])
    bart = -(n_b - 1 - (2 * k + 5) / 6) * np.linalg.slogdet(R)[1]
    df_b = k * (k - 1) / 2

    # Horn's parallel analysis on Pearson correlations of distinct patterns
    M = D["P_codes"][:, sym_idx]
    M = M[(M >= 0).all(axis=1)].astype(float)
    ev_obs = np.sort(np.linalg.eigvalsh(np.corrcoef(M, rowvar=False)))[::-1]
    sims = []
    for _ in range(500):
        Mp = np.column_stack([rng.permutation(M[:, i]) for i in range(M.shape[1])])
        sims.append(np.sort(np.linalg.eigvalsh(np.corrcoef(Mp, rowvar=False)))[::-1])
    ev_95 = np.percentile(np.array(sims), 95, axis=0)
    n_factors = int(np.argmax(ev_obs <= ev_95)) if (ev_obs <= ev_95).any() else len(ev_obs)

    out.update({
        "items": names, "ordinal_alpha": float(k * rbar / (1 + (k - 1) * rbar)),
        "mean_polychoric": float(rbar), "omega_total": omega, "loadings": lam.tolist(),
        "kmo": kmo_all, "kmo_item": kmo_item.tolist(),
        "bartlett_chi2": float(bart), "bartlett_df": int(df_b),
        "bartlett_p": float(stats.chi2.sf(bart, df_b)), "bartlett_n": n_b,
        "eig_obs": ev_obs.tolist(), "eig_95": ev_95.tolist(), "n_factors_parallel": n_factors,
    })
    return out


# ================================================================== S7 MCA
def s7_mca(D: dict) -> dict:
    codes = D["codes"].copy()
    for j in range(codes.shape[1]):                       # mode-impute the 27 cells
        col = codes[:, j]
        col[col < 0] = np.bincount(col[col >= 0]).argmax()
    K = D["K"]
    Q = len(K)
    offsets = np.concatenate([[0], np.cumsum(K)[:-1]])
    Z = np.zeros((len(codes), sum(K)))
    for j in range(Q):
        Z[np.arange(len(codes)), offsets[j] + codes[:, j]] = 1
    labels = [f"{SHORT[ITEMS[j]]}: {lv}" for j in range(Q) for lv in level_order(ITEMS[j])]
    item_of = [j for j in range(Q) for _ in range(K[j])]

    Pm = Z / Z.sum()
    r, c = Pm.sum(1), Pm.sum(0)
    S = (Pm - np.outer(r, c)) / np.sqrt(np.outer(r, c))
    Uu, s, Vt = np.linalg.svd(S, full_matrices=False)
    lam = s ** 2
    row_std = Uu / np.sqrt(r)[:, None]
    col_pc = (Vt.T / np.sqrt(c)[:, None]) * s

    # self-check: an active category's principal coordinate = mean row standard coordinate
    chk = row_std[Z[:, 0] == 1, :2].mean(0)
    assert np.allclose(chk, col_pc[0, :2], atol=1e-6), "MCA transition formula check failed"

    J = sum(K)
    adj = np.where(lam > 1 / Q, ((Q / (Q - 1)) * (lam - 1 / Q)) ** 2, 0.0)
    total_adj = (Q / (Q - 1)) * ((lam ** 2).sum() - (J - Q) / Q ** 2)
    y = D["y"]
    sup = np.array([row_std[y == k, :2].mean(0) for k in (0, 1)])
    row_pc = row_std * s
    eta2 = []
    for d in range(3):
        f = row_pc[:, d]
        ss_b = sum(((y == k).sum()) * (f[y == k].mean() - f.mean()) ** 2 for k in (0, 1))
        eta2.append(float(ss_b / ((f - f.mean()) ** 2).sum()))
    return {"labels": labels, "item_of": item_of, "col_pc": col_pc[:, :2], "sup": sup,
            "inertia_raw_pct": (100 * lam[:5] / lam.sum()).tolist(),
            "inertia_greenacre_pct": (100 * adj[:5] / total_adj).tolist(),
            "eta2_target": eta2}


# ============================================================ S8 missingness
def littles_mcar(Y: np.ndarray, iters: int = 1000, tol: float = 1e-8) -> dict:
    n, p = Y.shape
    M = np.isnan(Y)
    mu = np.nanmean(Y, 0)
    S = np.cov(np.where(M, mu, Y), rowvar=False) + 1e-6 * np.eye(p)
    comp, inc = ~M.any(1), np.flatnonzero(M.any(1))
    for _ in range(iters):
        T1 = Y[comp].sum(0)
        T2 = Y[comp].T @ Y[comp]
        for i in inc:
            o, m = ~M[i], M[i]
            B = S[np.ix_(m, o)] @ np.linalg.inv(S[np.ix_(o, o)])
            yi = Y[i].copy()
            yi[m] = mu[m] + B @ (Y[i, o] - mu[o])
            C = np.zeros((p, p))
            C[np.ix_(m, m)] = S[np.ix_(m, m)] - B @ S[np.ix_(o, m)]
            T1 = T1 + yi
            T2 = T2 + np.outer(yi, yi) + C
        mu_n = T1 / n
        S_n = T2 / n - np.outer(mu_n, mu_n)
        done = np.max(np.abs(mu_n - mu)) < tol and np.max(np.abs(S_n - S)) < tol
        mu, S = mu_n, S_n
        if done:
            break
    keys = [tuple(r) for r in M]
    d2, df = 0.0, 0
    for key in set(keys):
        idx = [i for i, k in enumerate(keys) if k == key]
        o = ~np.array(key)
        diff = Y[idx][:, o].mean(0) - mu[o]
        d2 += len(idx) * diff @ np.linalg.inv(S[np.ix_(o, o)]) @ diff
        df += int(o.sum())
    df -= p
    return {"d2": float(d2), "df": int(df),
            "p": float(stats.chi2.sf(d2, df)) if df > 0 else np.nan,
            "n_missing_patterns": len(set(keys)) - (1 if tuple([False] * p) in set(keys) else 0)}


def s8_missing(D: dict) -> dict:
    codes, g, y = D["codes"], D["g"], D["y"]
    miss_row = (codes < 0).any(1)
    per_item = {SHORT[c]: int((codes[:, j] < 0).sum()) for j, c in enumerate(ITEMS)}
    co = int(((codes < 0).sum(1) >= 2).sum())

    na_groups = pd.Series(g[miss_row]).value_counts()
    all_groups = pd.Series(g).value_counts()
    # complete copies: rows whose features equal an NA pattern's observed values elsewhere?
    pm = (D["P_codes"] < 0).any(1)
    t = np.array([[((~pm) & (D["P_lab"] == 0)).sum(), ((~pm) & (D["P_lab"] == 1)).sum()],
                  [(pm & (D["P_lab"] == 0)).sum(), (pm & (D["P_lab"] == 1)).sum()]])
    fisher_p = float(stats.fisher_exact(t)[1])

    Y = D["P_codes"].astype(float)
    Y[Y < 0] = np.nan
    little = littles_mcar(Y)
    return {"cells": int((codes < 0).sum()), "rows": int(miss_row.sum()),
            "rows_multi_missing": co, "per_item": per_item,
            "na_patterns": int(len(na_groups)),
            "na_pattern_row_counts": sorted(na_groups.tolist(), reverse=True),
            "na_pattern_total_sizes": sorted(all_groups[na_groups.index].tolist(), reverse=True),
            "fisher_table_patterns": t.tolist(), "fisher_p_patterns": fisher_p,
            "little": little}


# ========================================================== S9 duplication vs label
def s9_duplication(D: dict, rng) -> dict:
    """Is duplication label-neutral?

    The Mann-Whitney test on repeat counts is reported but is the wrong instrument: both
    labels have a median repeat count of 3, and the imbalance lives in the upper tail.
    The direct test permutes questionnaire labels among patterns (holding each pattern's
    repeat count fixed) and asks whether the observed row prevalence is extreme.
    """
    sizes, lab = D["P_size"], D["P_lab"]
    mw = stats.mannwhitneyu(sizes[lab == 1], sizes[lab == 0], alternative="two-sided")
    N = sizes.sum()
    pr = wilson(D["y"].sum(), len(D["y"]))
    pp = wilson(lab.sum(), len(lab))
    obs = float((sizes * lab).sum() / N)
    null = np.array([(sizes * rng.permutation(lab)).sum() / N for _ in range(20000)])
    p_perm = float((1 + (np.abs(null - null.mean()) >= abs(obs - null.mean())).sum()) / (1 + len(null)))
    return {
        "prev_rows_by_pattern_label": obs, "perm_null_mean": float(null.mean()),
        "perm_null_lo": float(np.percentile(null, 2.5)), "perm_null_hi": float(np.percentile(null, 97.5)),
        "perm_p": p_perm,
        "rows_from_neg": float(sizes[lab == 0].sum()), "rows_from_pos": float(sizes[lab == 1].sum()),
        "rows": int(N), "patterns": int(len(sizes)),
        "kish_n": float(N ** 2 / (sizes ** 2).sum()), "deff": float((sizes ** 2).sum() / N),
        "prev_rows": pr, "prev_patterns": pp,
        "mean_size_pos": float(sizes[lab == 1].mean()), "mean_size_neg": float(sizes[lab == 0].mean()),
        "median_size_pos": float(np.median(sizes[lab == 1])),
        "median_size_neg": float(np.median(sizes[lab == 0])),
        "mw_U": float(mw.statistic), "mw_p": float(mw.pvalue),
        "sizes_pos": sizes[lab == 1].tolist(), "sizes_neg": sizes[lab == 0].tolist(),
    }


# ==================================================================== figures
def _style(ax, grid_axis="both"):
    ax.grid(axis=grid_axis, alpha=0.25)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def fig_level_rates(levels: pd.DataFrame, prev: float, path: Path) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(14, 10.5))
    for ax, c in zip(axes.ravel(), ITEMS):
        sub = levels[levels["Item"] == c].reset_index(drop=True)
        yy = np.arange(len(sub))[::-1]
        ax.errorbar(sub["Target rate"], yy + 0.12,
                    xerr=[sub["Target rate"] - sub["Rate lo"], sub["Rate hi"] - sub["Target rate"]],
                    fmt="o", color=PAL["accent"], ms=6, capsize=3, lw=1.6, label="Rows (Wilson 95% CI)")
        ax.errorbar(sub["Pattern rate"], yy - 0.12,
                    xerr=[sub["Pattern rate"] - sub["Pattern lo"], sub["Pattern hi"] - sub["Pattern rate"]],
                    fmt="D", mfc="white", color=PAL["pat"], ms=5.5, capsize=3, lw=1.3,
                    label="Distinct patterns")
        ax.axvline(prev, color=PAL["grey"], ls="--", lw=1)
        ax.set_yticks(yy, [f"{l}  (n={r}, p={p})" for l, r, p in
                           zip(sub["Level"], sub["Rows"], sub["Patterns"])], fontsize=8)
        ax.set_xlim(0, 1)
        ax.set_title(SHORT[c], fontsize=10.5, loc="left")
        _style(ax, "x")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.supxlabel("Suicide-attempt rate  (dashed line = overall row prevalence)", fontsize=10)
    fig.suptitle("Target rate by response level — rows versus distinct questionnaires", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.legend(h, l, loc="upper right", fontsize=8.5, ncol=2, frameon=False,
               bbox_to_anchor=(0.995, 0.995))
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_effects(assoc: pd.DataFrame, path: Path) -> None:
    a = assoc.sort_values("V_bc")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    yy = np.arange(len(a))
    ax = axes[0]
    sig = a["q Holm (Rao-Scott)"] < 0.05
    colours = [PAL["accent"] if s else PAL["grey"] for s in sig]
    ax.barh(yy, a["V_bc"], color=colours, alpha=0.85, height=0.62)
    ax.errorbar(a["V_bc"], yy, xerr=[a["V_bc"] - a["V_lo"], a["V_hi"] - a["V_bc"]],
                fmt="none", ecolor=PAL["ink"], capsize=3, lw=1.2)
    ax.scatter(a["V_bc (patterns)"], yy, marker="D", s=34, facecolor="white",
               edgecolor=PAL["pat"], zorder=3, label="V on distinct patterns")
    for i, (_, r) in enumerate(a.iterrows()):
        ax.text(max(r["V_hi"], r["V_bc (patterns)"]) + 0.012, i,
                f"q={fmt_p(r['q Holm (Rao-Scott)'])}", va="center", fontsize=8, color=PAL["grey"])
    ax.set_yticks(yy, a["Short"])
    ax.set_xlabel("Bias-corrected Cramér's V  (bars: rows, 95% cluster-bootstrap CI)")
    ax.set_title("(a) Effect size — blue = significant after Rao–Scott + Holm", fontsize=10, loc="left")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_xlim(0, max(a["V_hi"].max(), a["V_bc (patterns)"].max()) + 0.12)
    _style(ax, "x")

    ax = axes[1]
    b = assoc.sort_values("IV")
    yb = np.arange(len(b))
    ax.barh(yb, b["IV"], color=PAL["accent"], alpha=0.85, height=0.62)
    for x, lbl in ((0.02, "weak"), (0.1, "medium"), (0.3, "strong"), (0.5, "suspicious")):
        ax.axvline(x, color=PAL["grey"], ls=":", lw=1)
        ax.text(x, 1.005, lbl, fontsize=7.5, color=PAL["grey"], ha="center", va="bottom",
                transform=ax.get_xaxis_transform())
    for i, v in enumerate(b["IV"]):
        ax.text(v + 0.006, i, f"{v:.3f}", va="center", fontsize=8)
    ax.set_yticks(yb, b["Short"])
    ax.set_xlabel("Information Value (weight-of-evidence)")
    ax.set_title("(b) Information Value with conventional bands", fontsize=10, loc="left", pad=16)
    _style(ax, "x")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_trend(trend: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), sharey=False)
    for ax, unit in zip(axes, ("rows", "patterns")):
        t = trend[trend["Unit"] == unit].sort_values("chi2_total")
        yy = np.arange(len(t))
        ax.barh(yy, t["chi2_trend"], color=PAL["accent"], height=0.6, label="Linear trend (1 df)")
        ax.barh(yy, t["chi2_dev"], left=t["chi2_trend"], color=PAL["pat"], height=0.6,
                label="Departure from linearity")
        for i, (_, r) in enumerate(t.iterrows()):
            ax.text(r["chi2_total"] * 1.01 + 0.5, i,
                    f"departure {100 * (1 - r['trend_share']):.0f}%  q={fmt_p(r['q Holm (dev)'])}",
                    va="center", fontsize=8, color=PAL["grey"])
        ax.set_yticks(yy, t["Short"])
        ax.set_xlabel("Pearson χ² partitioned (Cochran–Armitage)")
        ax.set_title(f"({'a' if unit == 'rows' else 'b'}) {unit.capitalize()}"
                     f"{' — naive, n=1,168' if unit == 'rows' else ' — one per questionnaire, n=248'}",
                     fontsize=10, loc="left")
        ax.set_xlim(0, t["chi2_total"].max() * 1.55)
        _style(ax, "x")
    axes[0].legend(fontsize=8, loc="lower right")
    fig.suptitle("How much of each item's association is a monotone severity trend?", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _heat(ax, M, labels, title, vmin, vmax, cmap, dark_above):
    Mm = np.ma.masked_invalid(M)
    cm = plt.get_cmap(cmap).copy()
    cm.set_bad("#eceff3")
    im = ax.imshow(Mm, vmin=vmin, vmax=vmax, cmap=cm)
    ax.set_xticks(range(len(labels)), labels, rotation=50, ha="right", fontsize=8)
    ax.set_yticks(range(len(labels)), labels, fontsize=8)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if np.isnan(M[i, j]):
                continue
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=6.5,
                    color="white" if abs(M[i, j]) > dark_above else PAL["ink"])
    ax.set_title(title, fontsize=10, loc="left")
    return im


def fig_dependence(dep: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(19, 6.4))
    L = dep["labels"]
    im = _heat(axes[0], dep["V"], L, "(a) Bias-corrected Cramér's V", 0, 0.6, "Blues", 0.35)
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.03)
    R = dep["polychoric"].copy()
    np.fill_diagonal(R, np.nan)
    im = _heat(axes[1], R, L, "(b) Polychoric / tetrachoric correlation", -0.8, 0.8, "RdBu_r", 0.5)
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.03)
    im = _heat(axes[2], dep["U"], L, "(c) Theil's U(row | column) — asymmetric", 0, 0.3, "Purples", 0.17)
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.03)
    fig.suptitle(f"Inter-item dependence (row-level; row-vs-pattern stability of V: "
                 f"r = {dep['stability_r']:.2f})", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_cluster_gvif(dep: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": [1.25, 1]})
    D_ = 1 - np.nan_to_num(dep["V"], nan=1.0)
    np.fill_diagonal(D_, 0)
    D_ = (D_ + D_.T) / 2
    Z = linkage(squareform(D_, checks=False), method="average")
    dendrogram(Z, labels=dep["labels"], ax=axes[0], color_threshold=0,
               above_threshold_color=PAL["accent"], leaf_rotation=40)
    axes[0].set_ylabel("1 − bias-corrected V  (average linkage)")
    axes[0].set_title("(a) Hierarchical clustering of items and target", fontsize=10, loc="left")
    _style(axes[0], "y")

    gv = dep["gvif"].sort_values("GVIF^(1/2df)")
    yy = np.arange(len(gv))
    axes[1].barh(yy, gv["GVIF^(1/2df)"], color=PAL["accent"], height=0.6)
    axes[1].axvline(np.sqrt(5), color=PAL["pat"], ls="--", lw=1.2, label="√5 (moderate)")
    axes[1].axvline(np.sqrt(10), color=PAL["ink"], ls=":", lw=1.2, label="√10 (severe)")
    for i, v in enumerate(gv["GVIF^(1/2df)"]):
        axes[1].text(v + 0.02, i, f"{v:.2f}", va="center", fontsize=8.5)
    axes[1].set_yticks(yy, gv["Short"])
    axes[1].set_xlabel("GVIF^(1/(2·df))  — comparable to √VIF")
    axes[1].set_title("(b) Multicollinearity of the one-hot design", fontsize=10, loc="left")
    axes[1].legend(fontsize=8, loc="lower right")
    axes[1].set_xlim(0.9, max(np.sqrt(10) + 0.3, gv["GVIF^(1/2df)"].max() + 0.3))
    _style(axes[1], "x")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_logit(lr: dict, path: Path) -> None:
    c = lr["coef"].iloc[1:].reset_index(drop=True)[::-1].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(10.5, 8.5))
    yy = np.arange(len(c))
    ax.hlines(yy, c["CI robust lo"], c["CI robust hi"], color=PAL["pat"], lw=4.5, alpha=0.55,
              label="95% CI, cluster-robust (by questionnaire)")
    ax.hlines(yy, c["CI naive lo"], c["CI naive hi"], color=PAL["ink"], lw=1.4,
              label="95% CI, naive (rows independent)")
    ax.scatter(c["OR"], yy, color=PAL["ink"], zorder=3, s=26)
    ax.scatter(c["OR (patterns)"], yy, marker="D", facecolor="white", edgecolor=PAL["accent"],
               zorder=4, s=30, label="OR fitted on distinct patterns")
    ax.axvline(1, color=PAL["grey"], ls="--", lw=1)
    ax.set_xscale("log")
    ax.set_yticks(yy, c["Term"], fontsize=8.5)
    ax.set_xlabel("Adjusted odds ratio (log scale)")
    ax.set_title(f"Multivariable logistic regression (n = {lr['n']}, {lr['n_clusters']} questionnaire "
                 f"clusters)\nmedian standard-error inflation under clustering ×"
                 f"{lr['median_se_inflation']:.2f}", fontsize=10.5, loc="left")
    ax.legend(fontsize=8.5, loc="upper right")
    _style(ax, "x")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_scale(scale: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    names = scale["items"]
    x = np.arange(len(names))
    ax = axes[0]
    ax.bar(x - 0.2, scale["rows"]["item_rest"], 0.4, color=PAL["accent"], label="Item–rest r (rows)")
    ax.bar(x + 0.2, scale["patterns"]["item_rest"], 0.4, color=PAL["pat"], label="Item–rest r (patterns)")
    ax.axhline(0.3, color=PAL["grey"], ls="--", lw=1, label="0.30 convention")
    ax.axhline(0, color=PAL["ink"], lw=0.8)
    ax2 = ax.twinx()
    ax2.plot(x, scale["rows"]["alpha_if_deleted"], "o", color=PAL["ink"], ms=6,
             label="α if item deleted (rows)")
    ax2.axhline(scale["rows"]["alpha"], color=PAL["ink"], ls=":", lw=1, label="α, all items (rows)")
    ax2.set_ylabel("Cronbach's α if deleted")
    ax.set_xticks(x, names, rotation=35, ha="right", fontsize=8.5)
    ax.set_ylabel("Corrected item–rest correlation")
    ax.set_title(f"(a) Item diagnostics — α rows {scale['rows']['alpha']:.2f}, "
                 f"patterns {scale['patterns']['alpha']:.2f}, ω {scale['omega_total']:.2f}",
                 fontsize=10, loc="left")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.26),
              ncol=3, frameon=False)
    _style(ax, "y")

    ax = axes[1]
    k = np.arange(1, len(scale["eig_obs"]) + 1)
    ax.plot(k, scale["eig_obs"], "o-", color=PAL["accent"], lw=2, label="Observed eigenvalues")
    ax.plot(k, scale["eig_95"], "s--", color=PAL["pat"], lw=1.6, label="95th pct, permuted data")
    ax.axhline(1, color=PAL["grey"], ls=":", lw=1, label="Kaiser = 1")
    ax.set_xlabel("Component")
    ax.set_ylabel("Eigenvalue")
    ax.set_title(f"(b) Horn's parallel analysis — retain {scale['n_factors_parallel']} "
                 f"(KMO {scale['kmo']:.2f})", fontsize=10, loc="left")
    ax.legend(fontsize=8.5)
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_mca(mca: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 8))
    cmap = plt.get_cmap("tab10")
    for i, (lbl, (x, y_)) in enumerate(zip(mca["labels"], mca["col_pc"])):
        j = mca["item_of"][i]
        ax.scatter(x, y_, color=cmap(j % 10), s=38, zorder=3)
        ax.annotate(lbl.split(": ")[1], (x, y_), textcoords="offset points", xytext=(4, 3),
                    fontsize=7.5, color=cmap(j % 10))
    for j, c in enumerate(ITEMS):
        ax.scatter([], [], color=cmap(j % 10), label=SHORT[c], s=30)
    for k, name in enumerate(("No attempt", "Attempt")):
        ax.scatter(*mca["sup"][k], marker="*", s=380, color=PAL["pat"] if k else PAL["ink"],
                   edgecolor="white", zorder=5)
        ax.annotate(f"TARGET: {name}", mca["sup"][k], textcoords="offset points", xytext=(8, -12),
                    fontsize=9.5, fontweight="bold", color=PAL["pat"] if k else PAL["ink"])
    ax.axhline(0, color=PAL["grey"], lw=0.8)
    ax.axvline(0, color=PAL["grey"], lw=0.8)
    gi = mca["inertia_greenacre_pct"]
    ax.set_xlabel(f"Dimension 1  ({gi[0]:.1f}% adjusted inertia; η² target {mca['eta2_target'][0]:.3f})")
    ax.set_ylabel(f"Dimension 2  ({gi[1]:.1f}% adjusted inertia; η² target {mca['eta2_target'][1]:.3f})")
    ax.set_title("Multiple correspondence analysis — response categories with the target as a "
                 "supplementary variable", fontsize=11, loc="left")
    ax.legend(fontsize=8, ncol=3, loc="lower left", title="Item", title_fontsize=8)
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_duplication(dup: dict, miss: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))
    ax = axes[0]
    rng = np.random.default_rng(1)
    for k, (key, name, colour) in enumerate((("sizes_neg", "No attempt", PAL["row"]),
                                             ("sizes_pos", "Attempt", PAL["pat"]))):
        v = np.array(dup[key])
        ax.scatter(k + rng.uniform(-0.18, 0.18, len(v)), v, s=12, alpha=0.55, color=colour)
        ax.boxplot(v, positions=[k], widths=0.45, showfliers=False,
                   medianprops={"color": PAL["ink"], "lw": 2})
    ax.set_xticks([0, 1], [f"No attempt\n({len(dup['sizes_neg'])} patterns)",
                           f"Attempt\n({len(dup['sizes_pos'])} patterns)"])
    ax.set_ylabel("Times the questionnaire is repeated")
    ax.set_title(f"(a) Repeat count by label — means {dup['mean_size_neg']:.1f} vs "
                 f"{dup['mean_size_pos']:.1f}, medians tied", fontsize=10, loc="left")
    _style(ax, "y")

    ax = axes[1]
    vals = [dup["prev_rows"], dup["prev_patterns"]]
    ax.bar([0, 1], [v[0] for v in vals], color=[PAL["row"], PAL["pat"]], width=0.55)
    ax.errorbar([0, 1], [v[0] for v in vals],
                yerr=[[v[0] - v[1] for v in vals], [v[2] - v[0] for v in vals]],
                fmt="none", ecolor=PAL["ink"], capsize=5)
    for i, v in enumerate(vals):
        ax.text(i, v[2] + 0.015, f"{100 * v[0]:.1f}%", ha="center", fontsize=10)
    ax.set_xticks([0, 1], [f"Rows\n(n={dup['rows']})", f"Distinct questionnaires\n(n={dup['patterns']})"])
    ax.set_ylim(0, 0.65)
    ax.set_ylabel("Positive prevalence (Wilson 95% CI)")
    ax.set_title(f"(b) Duplication favours negatives\npermutation p = {fmt_p(dup['perm_p'])}",
                 fontsize=10, loc="left")
    _style(ax, "y")

    ax = axes[2]
    labels = ["Rows", "Distinct\npatterns", "Kish\neffective n"]
    v = [dup["rows"], dup["patterns"], dup["kish_n"]]
    ax.bar(range(3), v, color=[PAL["row"], PAL["accent"], PAL["pat"]], width=0.6)
    for i, x in enumerate(v):
        ax.text(i, x + 18, f"{x:,.0f}", ha="center", fontsize=10)
    ax.set_xticks(range(3), labels)
    ax.set_ylabel("Sample size")
    ax.set_title(f"(c) Sample size under clustering — design effect {dup['deff']:.1f}",
                 fontsize=10, loc="left")
    ax.set_ylim(0, dup["rows"] * 1.12)
    _style(ax, "y")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ==================================================================== tables
def md_table(df: pd.DataFrame, cols: list[str], fmts: dict | None = None) -> str:
    fmts = fmts or {}
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            f = fmts.get(c)
            if f == "p":
                cells.append(fmt_p(v))
            elif f and isinstance(v, (int, float, np.floating, np.integer)) and not pd.isna(v):
                cells.append(format(v, f))
            elif isinstance(v, (float, np.floating)):
                cells.append("—" if pd.isna(v) else f"{v:.3f}")
            else:
                cells.append(str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, pd.DataFrame):
        return o.to_dict(orient="records")
    if isinstance(o, tuple):
        return list(o)
    raise TypeError(type(o))


# ===================================================================== main
def main() -> None:
    rng = np.random.default_rng(SEED)
    D = load()
    print(f"rows {len(D['y'])}, distinct questionnaires {len(D['P_size'])}", flush=True)

    print("S1 profiles", flush=True)
    prof, levels = s1_profiles(D)
    print("S2 association (bootstrap + permutation)", flush=True)
    assoc = s2_association(D, rng)
    comp = s2b_composite(D)
    print("S3 trend", flush=True)
    trend = s3_trend(D)
    print("S4 logistic regression", flush=True)
    lr = s4_logistic(D)
    print("S5 dependence (polychoric)", flush=True)
    dep = s5_dependence(D)
    print("S6 scale structure", flush=True)
    scale = s6_scale(D, dep, rng)
    print("S7 MCA", flush=True)
    mca = s7_mca(D)
    print("S8 missingness (Little's MCAR)", flush=True)
    miss = s8_missing(D)
    print("S9 duplication", flush=True)
    dup = s9_duplication(D, rng)

    # Robustness ladder: which item-target associations survive every correction?
    t = lr["terms"].set_index("Item")
    a = assoc.set_index("Item")
    tests_ = {
        "Naive χ² (rows)": a["p (naive rows)"],
        "Rao–Scott + Holm": a["q Holm (Rao-Scott)"],
        "Cluster permutation + Holm": a["q Holm (perm)"],
        "Distinct patterns + BH": a["q BH (patterns)"],
        "Adjusted, cluster-robust + Holm": t["q Holm robust"],
        "Adjusted, distinct patterns + Holm": t["q Holm LR (patterns)"],
    }
    ladder = pd.DataFrame({k: (v.reindex(ITEMS) < 0.05) for k, v in tests_.items()})
    ladder.insert(0, "Short", [SHORT[c] for c in ITEMS])
    ladder["Survives"] = ladder[list(tests_)].sum(1).astype(str) + f"/{len(tests_)}"
    ladder = ladder.reset_index(drop=True)

    print("figures", flush=True)
    fig_level_rates(levels, D["y"].mean(), OUT / "fig_stats_level_rates.png")
    fig_effects(assoc, OUT / "fig_stats_effect_sizes.png")
    fig_trend(trend, OUT / "fig_stats_trend.png")
    fig_dependence(dep, OUT / "fig_stats_dependence.png")
    fig_cluster_gvif(dep, OUT / "fig_stats_clustering_gvif.png")
    fig_logit(lr, OUT / "fig_stats_logistic.png")
    fig_scale(scale, OUT / "fig_stats_scale.png")
    fig_mca(mca, OUT / "fig_stats_mca.png")
    fig_duplication(dup, miss, OUT / "fig_stats_duplication.png")

    ladder.to_csv(OUT / "stats_robustness_ladder.csv", index=False)
    prof.to_csv(OUT / "stats_column_profile.csv", index=False)
    levels.to_csv(OUT / "stats_level_rates.csv", index=False)
    assoc.to_csv(OUT / "stats_univariate_association.csv", index=False)
    trend.to_csv(OUT / "stats_trend_tests.csv", index=False)
    lr["coef"].to_csv(OUT / "stats_logistic_coefficients.csv", index=False)
    lr["terms"].to_csv(OUT / "stats_logistic_terms.csv", index=False)
    dep["pairs"].to_csv(OUT / "stats_item_pair_tests.csv", index=False)
    dep["gvif"].to_csv(OUT / "stats_gvif.csv", index=False)
    for key in ("V", "V_patterns", "U", "polychoric"):
        pd.DataFrame(dep[key], index=dep["labels"], columns=dep["labels"]).to_csv(
            OUT / f"stats_matrix_{key.lower()}.csv")

    blob = {
        "composite_score": comp, "design_effect": assoc.attrs["deff"], "kish_n": assoc.attrs["kish_n"],
        "logistic": {k: v for k, v in lr.items() if k not in ("coef", "terms")},
        "dependence": {"stability_r": dep["stability_r"],
                       "pairs_sig_patterns_holm": int((dep["pairs"]["q Holm (patterns)"] < 0.05).sum()),
                       "pairs_sig_raoscott_holm": int((dep["pairs"]["q Holm (RS)"] < 0.05).sum()),
                       "n_pairs": int(len(dep["pairs"]))},
        "scale": scale, "mca": {k: v for k, v in mca.items() if k not in ("col_pc", "labels", "item_of")},
        "missing": miss, "duplication": {k: v for k, v in dup.items() if not k.startswith("sizes_")},
    }
    (OUT / "stats_summary.json").write_text(json.dumps(blob, indent=2, default=jsonable),
                                            encoding="utf-8")

    lad_md = ladder.copy()
    for k in tests_:
        lad_md[k] = lad_md[k].map({True: "✓", False: "·"})
    tables = [
        "## Robustness ladder — item–target association significant at 0.05?\n",
        md_table(lad_md, ["Short", *tests_, "Survives"]),
        "\n## S1 Column profile\n",
        md_table(prof, ["Short", "Levels", "Valid n", "Missing n", "Mode", "Mode share",
                        "Median level", "Entropy (bits)", "Norm. entropy", "Gini-Simpson",
                        "Imbalance (max/min)"],
                 {"Mode share": ".3f", "Entropy (bits)": ".3f", "Norm. entropy": ".3f",
                  "Gini-Simpson": ".3f", "Imbalance (max/min)": ".2f"}),
        "\n## S1b Level rates\n",
        md_table(levels, ["Short", "Level", "Rows", "Row %", "Patterns", "Target rate", "Rate lo",
                          "Rate hi", "Pattern rate"],
                 {"Row %": ".1f", "Target rate": ".3f", "Rate lo": ".3f", "Rate hi": ".3f",
                  "Pattern rate": ".3f"}),
        f"\n## S2 Univariate association (design effect {assoc.attrs['deff']:.2f}, "
        f"Kish n {assoc.attrs['kish_n']:.1f})\n",
        md_table(assoc, ["Short", "df", "Chi2 (rows)", "p (naive rows)", "Chi2 Rao-Scott",
                         "p Rao-Scott", "q Holm (Rao-Scott)", "Chi2 (patterns)", "p (patterns)",
                         "q BH (patterns)"],
                 {"Chi2 (rows)": ".1f", "Chi2 Rao-Scott": ".2f", "Chi2 (patterns)": ".2f",
                  "p (naive rows)": "p", "p Rao-Scott": "p", "q Holm (Rao-Scott)": "p",
                  "p (patterns)": "p", "q BH (patterns)": "p"}),
        "\n",
        md_table(assoc, ["Short", "V_bc", "V_lo", "V_hi", "V_bc (patterns)", "MI (bits)",
                         "U(target|item)", "p perm (cluster)", "q Holm (perm)", "IV"],
                 {"p perm (cluster)": "p", "q Holm (perm)": "p", "MI (bits)": ".4f",
                  "U(target|item)": ".4f"}),
        "\n## S3 Trend / departure from linearity\n",
        md_table(trend, ["Short", "Unit", "K", "z", "p_trend", "chi2_trend", "chi2_dev", "df_dev",
                         "p_dev", "q Holm (dev)", "trend_share"],
                 {"z": ".2f", "p_trend": "p", "chi2_trend": ".2f", "chi2_dev": ".2f", "p_dev": "p",
                  "q Holm (dev)": "p", "trend_share": ".2f"}),
        "\n## S4 Logistic regression — terms\n",
        md_table(lr["terms"], [c for c in ["Short", "df", "LR chi2 (naive)", "p LR (naive)",
                                            "Wald chi2 (robust)", "p Wald (robust)", "q Holm robust",
                                            "Linearity Wald (robust)", "p linearity (robust)",
                                            "q Holm linearity"] if c in lr["terms"]],
                 {"LR chi2 (naive)": ".2f", "p LR (naive)": "p", "Wald chi2 (robust)": ".2f",
                  "p Wald (robust)": "p", "q Holm robust": "p", "Linearity Wald (robust)": ".2f",
                  "p linearity (robust)": "p", "q Holm linearity": "p"}),
        "\n## S4b Logistic regression — coefficients\n",
        md_table(lr["coef"].iloc[1:], ["Term", "OR", "CI naive lo", "CI naive hi", "CI robust lo",
                                       "CI robust hi", "p naive", "p robust", "SE inflation",
                                       "OR (patterns)"],
                 {"OR": ".2f", "CI naive lo": ".2f", "CI naive hi": ".2f", "CI robust lo": ".2f",
                  "CI robust hi": ".2f", "p naive": "p", "p robust": "p", "SE inflation": ".2f",
                  "OR (patterns)": ".2f"}),
        "\n## S5 GVIF\n",
        md_table(dep["gvif"], ["Short", "df", "GVIF", "GVIF^(1/2df)"],
                 {"GVIF": ".2f", "GVIF^(1/2df)": ".2f"}),
    ]
    (OUT / "stats_tables.md").write_text("\n".join(tables), encoding="utf-8")

    # ---------------------------------------------------------------- console digest
    pd.set_option("display.width", 220)
    print("\n=== Robustness ladder (True = significant at 0.05) ===")
    print(ladder.to_string(index=False))
    print("\n=== S1 profile ===")
    print(prof[["Short", "Levels", "Missing n", "Mode", "Mode share", "Norm. entropy",
                "Gini-Simpson", "Imbalance (max/min)"]].round(3).to_string(index=False))
    print(f"\n=== S2 association  (deff {assoc.attrs['deff']:.2f}, Kish n {assoc.attrs['kish_n']:.1f}) ===")
    print(assoc[["Short", "Chi2 (rows)", "p (naive rows)", "Chi2 Rao-Scott", "q Holm (Rao-Scott)",
                 "p (patterns)", "q BH (patterns)", "V_bc", "V_lo", "V_hi", "V_bc (patterns)",
                 "p perm (cluster)", "IV"]].round(4).to_string(index=False))
    print("\ncomposite score:", {k: round(v, 4) for k, v in comp.items()})
    print("\n=== S3 trend ===")
    print(trend[["Short", "Unit", "z", "p_trend", "chi2_dev", "p_dev", "q Holm (dev)",
                 "trend_share"]].round(4).to_string(index=False))
    print(f"\n=== S4 logistic  McFadden R2 {lr['mcfadden_r2']:.3f} (patterns {lr['mcfadden_r2_patterns']:.3f}),"
          f" median SE inflation x{lr['median_se_inflation']:.2f} ===")
    print(lr["terms"].round(4).to_string(index=False))
    print("\n=== S5 dependence ===")
    print(f"row-vs-pattern stability r = {dep['stability_r']:.3f};  item pairs significant: "
          f"patterns+Holm {blob['dependence']['pairs_sig_patterns_holm']}/{len(dep['pairs'])}, "
          f"Rao-Scott+Holm {blob['dependence']['pairs_sig_raoscott_holm']}/{len(dep['pairs'])}")
    print(dep["gvif"].round(3).to_string(index=False))
    print("\n=== S6 scale ===")
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in scale.items()
           if k in ("ordinal_alpha", "mean_polychoric", "omega_total", "kmo", "bartlett_chi2",
                    "bartlett_p", "n_factors_parallel")},
          "| alpha rows", round(scale["rows"]["alpha"], 3), "patterns", round(scale["patterns"]["alpha"], 3))
    print("item-rest (rows):", [round(v, 3) for v in scale["rows"]["item_rest"]])
    print("loadings:", [round(v, 3) for v in scale["loadings"]])
    print("\n=== S7 MCA ===")
    print("Greenacre %:", [round(v, 1) for v in mca["inertia_greenacre_pct"]],
          "| eta2 target:", [round(v, 4) for v in mca["eta2_target"]])
    print("\n=== S8 missing ===")
    print({k: v for k, v in miss.items() if k != "little"}, "| Little:", miss["little"])
    print("\n=== S9 duplication ===")
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in dup.items()
           if not k.startswith("sizes_")})
    print(f"\nwritten to {OUT}")


if __name__ == "__main__":
    main()
