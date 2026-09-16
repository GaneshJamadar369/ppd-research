"""
Depression risk model on the pooled NHANES cohort (run nhanes_build.py first).

Target: PHQ-9 >= 10. Predictors come only from instruments other than the PHQ-9 — demographics,
reproductive history, blood work, BMI, sleep — so the label is never a function of the features.

Two populations are fitted:
  ALL       women 18-49 (larger, better powered)
  PERINATAL pregnant now or <= 24 months postpartum

    python nhanes_model.py
"""

import os

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 4))

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
FEATURES = ["RIDAGEYR", "RIDRETH1", "DMDEDUC2", "DMDMARTL", "INDFMPIR",   # demographics
            "RHQ131", "RHD143", "RHQ200", "RHD180", "months_pp",          # reproductive
            "LBXHGB", "LBXRBCSI", "LBXWBCSI", "LBXPLTSI", "LBXHSCRP",     # blood work
            "BMXBMI", "SLD010H", "SLD012",                                # body / sleep
            "pregnant_now", "postpartum_12m"]


def boot_ci(y, p, n=1000):
    rng = np.random.default_rng(SEED)
    v = [roc_auc_score(y[i], p[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(n))
         if len(np.unique(y[i])) == 2]
    return np.percentile(v, 2.5), np.percentile(v, 97.5)


def run(d, label):
    X = d[[c for c in FEATURES if c in d.columns]].astype(float)
    y = d["depressed"].astype(int).to_numpy()
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)

    models = {
        "Gradient Boosting": HistGradientBoostingClassifier(
            max_leaf_nodes=15, min_samples_leaf=25, learning_rate=0.05, random_state=SEED),
        "Random Forest": make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(n_estimators=400, min_samples_leaf=10,
                                   class_weight="balanced", random_state=SEED, n_jobs=-1)),
        "Logistic Regression": make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED)),
        "Baseline (majority)": DummyClassifier(strategy="prior"),
    }

    rows = []
    for name, m in models.items():
        p = cross_val_predict(m, X, y, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
        auc = roc_auc_score(y, p)
        lo, hi = boot_ci(y, p)
        fpr, tpr, thr = roc_curve(y, p)
        j = np.argmax(tpr - fpr)
        rows.append({"Population": label, "Model": name, "n": len(y), "Prevalence": y.mean(),
                     "AUC": auc, "AUC lo": lo, "AUC hi": hi,
                     "Sensitivity": tpr[j], "Specificity": 1 - fpr[j]})
    return pd.DataFrame(rows).sort_values("AUC", ascending=False)


def main():
    d = pd.read_csv("nhanes_perinatal.csv")
    for c in ("pregnant_now", "postpartum_12m", "perinatal"):
        d[c] = d[c].astype(bool)

    res = pd.concat([run(d, "All women 18-49"),
                     run(d[d.perinatal], "Perinatal only")], ignore_index=True)
    res.insert(0, "Rank", res.groupby("Population")["AUC"].rank(ascending=False).astype(int))
    res = res.sort_values(["Population", "AUC"], ascending=[True, False])
    res.to_csv("results/nhanes_model_comparison.csv", index=False)
    pd.set_option("display.width", 200)
    print(res.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
