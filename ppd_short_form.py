"""
Short-form perinatal depression screener.

The 9 PHQ-9 items sum to `Scalling`, and `Labelling` is exactly `Scalling >= 10`, so feeding all
9 items to a model is arithmetic, not prediction. The legitimate question this data can answer is:
how few questions are needed to recover PHQ-9 caseness?

Item subsets and cut-offs are chosen on the training split only; every number reported is on the
held-out test split.

    python ppd_short_form.py
"""

import os
from itertools import combinations

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 4))

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

SEED = 42
DATA = "phq9_perinatal.csv"
OUT = "results/short_form_comparison.csv"

ITEM_NAMES = ["Interest", "Down", "Sleep", "Tired", "Appetite", "Failure", "Concentration",
              "Slowness", "Self-harm"]


def load():
    d = pd.read_csv(DATA)
    d.columns = [c.strip() for c in d.columns]
    for c in d.select_dtypes("object"):
        d[c] = d[c].str.strip()
    n_raw = len(d)
    d = d.drop_duplicates().reset_index(drop=True)
    socio = list(d.columns[:16])
    items = list(d.columns[16:25])
    d[items] = d[items].fillna(0)
    y = (d["Labelling"] == "Depressed").astype(int).to_numpy()
    print(f"{n_raw} rows -> {len(d)} unique | positive rate {y.mean():.3f}")
    return d, socio, items, y


def youden_cut(y, score):
    fpr, tpr, thr = roc_curve(y, score)
    return thr[np.argmax(tpr - fpr)]


def evaluate(y, score, cut):
    pred = (score >= cut).astype(int)
    tp = ((pred == 1) & (y == 1)).sum()
    tn = ((pred == 0) & (y == 0)).sum()
    fp = ((pred == 1) & (y == 0)).sum()
    fn = ((pred == 0) & (y == 1)).sum()
    rng = np.random.default_rng(SEED)
    boots = []
    for _ in range(1000):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) == 2:
            boots.append(roc_auc_score(y[i], score[i]))
    return {
        "AUC": roc_auc_score(y, score),
        "AUC lo": np.percentile(boots, 2.5),
        "AUC hi": np.percentile(boots, 97.5),
        "Sensitivity": tp / (tp + fn),
        "Specificity": tn / (tn + fp),
        "PPV": tp / (tp + fp) if tp + fp else 0.0,
        "F1": f1_score(y, pred),
        "Accuracy": (tp + tn) / len(y),
    }


def main():
    d, socio, items, y = load()
    tr, te = train_test_split(np.arange(len(d)), test_size=0.2, stratify=y, random_state=SEED)
    Xi = d[items].to_numpy()
    rows = []

    def add(name, n_q, score_tr, score_te, cut=None):
        cut = youden_cut(y[tr], score_tr) if cut is None else cut
        rows.append({"Model": name, "Questions": n_q, "Cut-off": round(float(cut), 3),
                     **evaluate(y[te], score_te, cut)})

    # Standard clinical short form: PHQ-2 (items 1+2), cut-off >= 3
    phq2 = Xi[:, 0] + Xi[:, 1]
    add("PHQ-2 (standard, cut-off 3)", 2, phq2[tr], phq2[te], cut=3)

    # Best item subsets by sum score, selected on train only
    for k in (2, 3, 4):
        best = max(combinations(range(9), k),
                   key=lambda s: roc_auc_score(y[tr], Xi[tr][:, list(s)].sum(1)))
        s = Xi[:, list(best)].sum(1)
        add(f"Best {k} items: " + " + ".join(ITEM_NAMES[i] for i in best), k, s[tr], s[te])
        if k == 3:
            best3 = best

    # Best 3 items + sociodemographics, gradient boosting
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    S = enc.fit_transform(d[socio].astype(str))
    X3s = np.hstack([Xi[:, list(best3)], S])
    gb = HistGradientBoostingClassifier(random_state=SEED).fit(X3s[tr], y[tr])
    p = gb.predict_proba(X3s)[:, 1]
    add("Best 3 items + sociodemographics (GBM)", 3, p[tr], p[te])

    # Reference: sociodemographics alone (no questions asked)
    gb0 = HistGradientBoostingClassifier(random_state=SEED).fit(S[tr], y[tr])
    p0 = gb0.predict_proba(S)[:, 1]
    add("Sociodemographics only (no questions)", 0, p0[tr], p0[te])

    res = pd.DataFrame(rows)
    # Rank: highest AUC first; among near-ties prefer fewer questions
    res = res.sort_values(["AUC", "Questions"], ascending=[False, True]).reset_index(drop=True)
    res.insert(0, "Rank", range(1, len(res) + 1))
    res.to_csv(OUT, index=False)
    pd.set_option("display.width", 200)
    print(res.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
