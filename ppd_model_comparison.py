"""
Postpartum Depression (PPD) — structured/clinical branch baseline comparison.

Dataset : post natal data.csv  (Kaggle "PostPartum Depression" survey, 1,503 responses)
Target  : 'Suicide attempt'  — Yes/No binary (the PPD severity proxy used in prior work).
          'Not interested to say' (n=335) is treated as non-response and dropped.
Features: Age bracket + 8 categorical symptom items (see ppd_preprocessing.py, which
          owns cleaning, canonicalisation, the engineered severity score and encoding).

Models  : Logistic Regression, SVM (RBF), Random Forest, ANN (MLP)
Metrics : Accuracy, Precision, Recall, F1, ROC-AUC, Confusion matrix (+ Specificity, MCC)

Two evaluation protocols are reported, because the 1,168 analysed rows contain only 248
unique response patterns, of which just one carries conflicting labels:

  Protocol A — Standard stratified 5-fold CV over all rows. Comparable to published
               results on this dataset, but duplicated respondents span train and test,
               so the scores are optimistically biased.
  Protocol B — Leakage-controlled: StratifiedGroupKFold with the group = the unique
               questionnaire response pattern, so identical rows never appear in both
               train and test. This is the defensible generalisation estimate.

Outputs land in ./results/.
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedGroupKFold,
    StratifiedKFold,
    cross_val_predict,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from ppd_preprocessing import (
    NON_RESPONSE,
    SCORE_COL,
    TARGET,
    build_preprocessor,
    prepare,
)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

RANDOM_STATE = 42
N_SPLITS = 5
HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
OUT.mkdir(exist_ok=True)

# Set by main() from --encoding; consumed by build_models / plot_importance.
ENCODING = "onehot"


# ------------------------------------------------------------------------- models
def build_models() -> dict[str, tuple[Pipeline, dict]]:
    """name -> (pipeline, hyperparameter grid tuned by inner CV on the training fold).

    Every pipeline starts with the fold-safe preprocessor from `ppd_preprocessing`,
    so imputation, encoding and scaling are all fitted on training folds only.
    """
    return {
        "Logistic Regression": (
            Pipeline(
                [
                    ("prep", build_preprocessor(ENCODING, scale=True)),
                    ("clf", LogisticRegression(max_iter=5000, random_state=RANDOM_STATE)),
                ]
            ),
            {"clf__C": [0.01, 0.1, 1.0, 10.0], "clf__class_weight": [None, "balanced"]},
        ),
        "SVM (RBF)": (
            Pipeline(
                [
                    ("prep", build_preprocessor(ENCODING, scale=True)),
                    ("clf", SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE)),
                ]
            ),
            {"clf__C": [0.1, 1.0, 10.0], "clf__gamma": ["scale", 0.01, 0.1]},
        ),
        "Random Forest": (
            Pipeline(
                [
                    ("prep", build_preprocessor(ENCODING, scale=False)),
                    (
                        "clf",
                        RandomForestClassifier(
                            n_estimators=500, random_state=RANDOM_STATE, n_jobs=-1
                        ),
                    ),
                ]
            ),
            {
                "clf__max_depth": [None, 6, 12],
                "clf__min_samples_leaf": [1, 3, 10],
            },
        ),
        "ANN (MLP)": (
            Pipeline(
                [
                    ("prep", build_preprocessor(ENCODING, scale=True)),
                    (
                        "clf",
                        MLPClassifier(
                            max_iter=1500,
                            early_stopping=True,
                            n_iter_no_change=25,
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            ),
            {
                "clf__hidden_layer_sizes": [(32,), (64, 32), (128, 64)],
                "clf__alpha": [1e-4, 1e-2],
                "clf__learning_rate_init": [1e-3, 1e-2],
            },
        ),
    }


# ---------------------------------------------------------------------- evaluation
def fold_metrics(y_true, y_pred, y_prob) -> dict[str, float]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_true, y_prob),
        "Specificity": tn / (tn + fp) if (tn + fp) else 0.0,
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


def pick_threshold(estimator, X_tr, y_tr, g_tr, cv) -> float:
    """Choose the decision threshold that maximises F1, using training-fold CV only.

    The default 0.5 cut-off is poorly placed for the imbalanced, heavily tied
    probabilities this survey produces (SVM in particular trades almost all recall
    for precision), which is the wrong bias for a screening instrument.
    """
    kwargs = {"groups": g_tr} if g_tr is not None else {}
    prob = cross_val_predict(estimator, X_tr, y_tr, cv=cv, method="predict_proba",
                             n_jobs=-1, **kwargs)[:, 1]
    grid = np.unique(np.round(np.linspace(0.05, 0.95, 91), 3))
    scores = [f1_score(y_tr, (prob >= t).astype(int), zero_division=0) for t in grid]
    return float(grid[int(np.argmax(scores))])


def run_protocol(
    name: str,
    X: pd.DataFrame,
    y: pd.Series,
    groups: np.ndarray | None,
) -> tuple[pd.DataFrame, dict]:
    """Nested CV: outer folds score, inner folds tune. Returns (summary table, artefacts)."""
    grouped = groups is not None
    if grouped:
        outer = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
        inner = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    else:
        outer = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
        inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

    models = build_models()
    rows, artefacts = [], {}
    yv = y.to_numpy()

    print(f"\n{'=' * 78}\n{name}\n{'=' * 78}")
    tuned_rows = []
    for model_name, (pipe, grid) in models.items():
        per_fold, tuned_fold, chosen, thresholds = [], [], [], []
        oof_pred = np.zeros(len(y))
        oof_prob = np.zeros(len(y))
        oof_tuned = np.zeros(len(y))

        splits = outer.split(X, yv, groups) if grouped else outer.split(X, yv)
        for tr, te in splits:
            search = GridSearchCV(pipe, grid, scoring="roc_auc", cv=inner, n_jobs=-1, refit=True)
            g_tr = groups[tr] if grouped else None
            if grouped:
                search.fit(X.iloc[tr], yv[tr], groups=g_tr)
            else:
                search.fit(X.iloc[tr], yv[tr])

            best = search.best_estimator_
            chosen.append(search.best_params_)
            oof_prob[te] = best.predict_proba(X.iloc[te])[:, 1]
            oof_pred[te] = best.predict(X.iloc[te])
            per_fold.append(fold_metrics(yv[te], oof_pred[te], oof_prob[te]))

            thr = pick_threshold(best, X.iloc[tr], yv[tr], g_tr, inner)
            thresholds.append(thr)
            oof_tuned[te] = (oof_prob[te] >= thr).astype(int)
            tuned_fold.append(fold_metrics(yv[te], oof_tuned[te], oof_prob[te]))

        folds = pd.DataFrame(per_fold)
        row = {"Model": model_name}
        for metric in folds.columns:
            row[metric] = folds[metric].mean()
            row[f"{metric} (SD)"] = folds[metric].std(ddof=1)
        cm = confusion_matrix(yv, oof_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        row.update({"TN": tn, "FP": fp, "FN": fn, "TP": tp})
        rows.append(row)

        tfolds = pd.DataFrame(tuned_fold)
        trow = {"Model": model_name}
        for metric in tfolds.columns:
            trow[metric] = tfolds[metric].mean()
            trow[f"{metric} (SD)"] = tfolds[metric].std(ddof=1)
        tcm = confusion_matrix(yv, oof_tuned, labels=[0, 1])
        tn, fp, fn, tp = tcm.ravel()
        trow.update({"TN": tn, "FP": fp, "FN": fn, "TP": tp,
                     "Threshold": float(np.mean(thresholds))})
        tuned_rows.append(trow)

        artefacts[model_name] = {
            "cm": cm,
            "oof_prob": oof_prob,
            "per_fold": folds,
            "best_params": chosen,
            "thresholds": thresholds,
        }
        print(
            f"  {model_name:<22} "
            f"Acc {row['Accuracy']:.3f}±{row['Accuracy (SD)']:.3f}  "
            f"Prec {row['Precision']:.3f}  Rec {row['Recall']:.3f}  "
            f"F1 {row['F1']:.3f}  AUC {row['ROC-AUC']:.3f}"
            f"   | tuned thr {trow['Threshold']:.2f}: "
            f"Rec {trow['Recall']:.3f}  F1 {trow['F1']:.3f}"
        )

    table = pd.DataFrame(rows).sort_values("ROC-AUC", ascending=False).reset_index(drop=True)
    tuned = (
        pd.DataFrame(tuned_rows).sort_values("F1", ascending=False).reset_index(drop=True)
    )
    artefacts["__tuned__"] = tuned
    return table, artefacts


# -------------------------------------------------------------------------- output
METRIC_ORDER = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "Specificity", "MCC"]


def to_markdown(table: pd.DataFrame, title: str, note: str) -> str:
    has_thr = "Threshold" in table.columns
    lines = [f"### {title}", "", note, ""]
    extra = ["Threshold"] if has_thr else []
    header = "| Model | " + " | ".join(METRIC_ORDER + extra) + " | TN | FP | FN | TP |"
    lines += [header, "|" + "---|" * (len(METRIC_ORDER) + len(extra) + 5)]
    for _, r in table.iterrows():
        cells = [f"{r[m]:.4f} ± {r[m + ' (SD)']:.3f}" for m in METRIC_ORDER]
        if has_thr:
            cells.append(f"{r['Threshold']:.2f}")
        counts = [str(int(r[c])) for c in ("TN", "FP", "FN", "TP")]
        lines.append("| " + " | ".join([r["Model"]] + cells + counts) + " |")
    return "\n".join(lines) + "\n"


def model_artefacts(artefacts: dict) -> dict:
    """Drop bookkeeping entries (e.g. the tuned-threshold table) from the artefact map."""
    return {k: v for k, v in artefacts.items() if not k.startswith("__")}


def plot_roc(y, artefacts, title, path):
    plt.figure(figsize=(6.5, 5.5))
    for model_name, art in model_artefacts(artefacts).items():
        fpr, tpr, _ = roc_curve(y, art["oof_prob"])
        plt.plot(fpr, tpr, lw=2, label=f"{model_name} (AUC = {auc(fpr, tpr):.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1, label="Chance")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title(title)
    plt.legend(loc="lower right", fontsize=9)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_confusions(artefacts, title, path):
    arts = model_artefacts(artefacts)
    fig, axes = plt.subplots(1, len(arts), figsize=(4 * len(arts), 3.8))
    for ax, (model_name, art) in zip(np.atleast_1d(axes), arts.items()):
        cm = art["cm"]
        ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(
                    j, i, f"{cm[i, j]}", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=13,
                )
        ax.set_xticks([0, 1], ["Pred No", "Pred Yes"])
        ax.set_yticks([0, 1], ["True No", "True Yes"])
        ax.set_title(model_name, fontsize=10)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_importance(X, y, path):
    """Random-Forest impurity importance, fit on the full dataset (descriptive only)."""
    rf = Pipeline(
        [
            ("prep", build_preprocessor(ENCODING, scale=False)),
            ("clf", RandomForestClassifier(n_estimators=500, random_state=RANDOM_STATE, n_jobs=-1)),
        ]
    ).fit(X, y)
    names = [
        n.split("__", 1)[-1] if n.startswith(("nominal__", "ordinal__", "engineered__")) else n
        for n in rf.named_steps["prep"].get_feature_names_out()
    ]
    imp = pd.Series(rf.named_steps["clf"].feature_importances_, index=names).nlargest(15)[::-1]

    plt.figure(figsize=(8.5, 6))
    plt.barh(imp.index, imp.values, color="#3b6ea5")
    plt.xlabel("Mean decrease in impurity")
    plt.title(f"Random Forest — top 15 features ({ENCODING} encoding)")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    return imp[::-1]


# ---------------------------------------------------------------------------- main
def main() -> None:
    global ENCODING
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--encoding", default="onehot", choices=["onehot", "ordinal", "hybrid"],
        help="Feature representation from ppd_preprocessing (default: onehot).",
    )
    args = parser.parse_args()
    ENCODING = args.encoding

    prep = prepare()
    X, y, groups = prep["X"], prep["y"], prep["groups"]
    print(f"Encoding                    : {ENCODING}")

    table_a, art_a = run_protocol(
        "PROTOCOL A — standard stratified 5-fold CV (literature-comparable, duplicate-inflated)",
        X, y, None,
    )
    table_b, art_b = run_protocol(
        "PROTOCOL B — leakage-controlled StratifiedGroupKFold (groups = unique response pattern)",
        X, y, groups,
    )

    tuned_b = art_b["__tuned__"]
    table_a.to_csv(OUT / "comparison_protocolA_standard.csv", index=False)
    table_b.to_csv(OUT / "comparison_protocolB_grouped.csv", index=False)
    tuned_b.to_csv(OUT / "comparison_protocolB_tuned_threshold.csv", index=False)

    n_patterns = len(np.unique(groups))
    md = [
        "# PPD Classifier Comparison — `post natal data.csv`",
        "",
        f"**Target:** `{TARGET}` (Yes = 1, No = 0); `{NON_RESPONSE}` responses dropped.  ",
        f"**Samples:** {len(y)} | **Positive rate:** {y.mean():.3f} | "
        f"**Encoding:** `{ENCODING}` (+ engineered `{SCORE_COL}`)  ",
        "**Preprocessing:** see `preprocessing_report.md` — category canonicalisation is "
        "applied up front; imputation, encoding and scaling are fitted inside each training "
        "fold only.  ",
        f"**Evaluation:** nested CV — outer {N_SPLITS}-fold scoring, inner 3-fold "
        "grid search on ROC-AUC. Mean ± SD across outer folds; "
        "confusion-matrix counts are pooled out-of-fold predictions.",
        "",
        to_markdown(
            table_a,
            "Protocol A — standard stratified 5-fold CV",
            f"_Comparable to published results on this dataset. These {len(y)} rows contain "
            f"only {n_patterns} unique response patterns, so near-identical rows fall on both "
            "sides of the split and these scores are optimistically biased._",
        ),
        "",
        to_markdown(
            table_b,
            "Protocol B — leakage-controlled grouped 5-fold CV",
            "_All rows sharing an identical questionnaire response pattern are kept in the "
            "same fold. This is the honest generalisation estimate and the one to report as "
            "the primary result._",
        ),
        "",
        to_markdown(
            tuned_b,
            "Protocol B (supplementary) — decision threshold tuned for F1",
            "_Same grouped folds, but the operating point is selected on the training folds "
            "instead of fixed at 0.5. Reported because a screening instrument should not "
            "trade away recall: at 0.5 the RBF-SVM reaches high precision while missing most "
            "positive cases. `Threshold` is the mean cut-off chosen across outer folds._",
        ),
    ]
    (OUT / "comparison_table.md").write_text("\n".join(md), encoding="utf-8")

    plot_roc(y, art_a, "ROC — Protocol A (standard CV)", OUT / "roc_protocolA.png")
    plot_roc(y, art_b, "ROC — Protocol B (grouped CV)", OUT / "roc_protocolB.png")
    plot_confusions(art_a, "Confusion matrices — Protocol A (pooled out-of-fold)",
                    OUT / "confusion_protocolA.png")
    plot_confusions(art_b, "Confusion matrices — Protocol B (pooled out-of-fold)",
                    OUT / "confusion_protocolB.png")
    imp = plot_importance(X, y, OUT / "feature_importance.png")

    (OUT / "selected_hyperparameters.json").write_text(
        json.dumps(
            {
                "protocol_A": {
                    m: a["best_params"] for m, a in model_artefacts(art_a).items()
                },
                "protocol_B": {
                    m: {"params": a["best_params"], "tuned_thresholds": a["thresholds"]}
                    for m, a in model_artefacts(art_b).items()
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\nTop predictive features (RF, full data):")
    for k, v in imp.head(8).items():
        print(f"  {k:<55} {v:.4f}")
    print(f"\nArtefacts written to {OUT}")


if __name__ == "__main__":
    main()
